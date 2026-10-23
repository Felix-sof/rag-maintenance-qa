"""
LLM generation with Google Gemini (free tier), with retries for transient errors.

The API key is read from GEMINI_API_KEY (via .env).
"""

import os
import re
import ssl
import time

import certifi
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
MAX_RETRIES = 5  # for API errors (429/5xx): each waits seconds to tens of seconds
MAX_NETWORK_RETRIES = 12  # for dropped TLS connections: usually transient, retried after 1s
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_WAIT_SECONDS = 60  # a 429 asking us to wait longer than this is a daily quota, not a blip
MAX_TOOL_ROUNDS = 1  # rounds of tool calls before the model must answer: at most 2 LLM calls a question
MAX_CALLS_PER_ROUND = 3


def retry_delay_seconds(message: str) -> float:
    """Parse Gemini's "Please retry in 6h51m33.5s" hint (0 if absent)."""
    match = re.search(r"retry in ((?:\d+h)?(?:\d+m)?(?:[\d.]+s)?)", message or "")
    if not match:
        return 0.0
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)([hms])", match.group(1)):
        total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


class GenerationError(Exception):
    """The LLM call failed after retries (quota, network, invalid key, ...)."""


def _new_client() -> genai.Client:
    """Gemini client that opens a fresh TLS connection per request.

    Disabling TLS session tickets and keep-alive avoids a pooled connection
    poisoned by antivirus / TLS-inspecting software (seen as an intermittent
    "SSL: INVALID_SESSION_ID" error on Windows) being reused across retries.
    Certificate verification is unaffected.
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.options |= ssl.OP_NO_TICKET
    client_args = {"verify": ctx, "limits": httpx.Limits(max_keepalive_connections=0)}
    return genai.Client(http_options=types.HttpOptions(client_args=client_args))


class GeminiGenerator:
    def __init__(self, model: str = MODEL):
        self.model = model
        self._client = None

    def generate(self, system: str, prompt: str, temperature: float = 0.2, usage: dict = None) -> str:
        """The model's answer to one prompt.

        If `usage` is given, the call is counted in it: "llm_calls", and the
        "prompt_tokens" / "output_tokens" (output includes a thinking model's
        thought tokens).
        """
        config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
        return (self._complete(prompt, config, usage).text or "").strip()

    def generate_with_tools(
        self, system: str, prompt: str, tools: list[dict], call_tool, temperature: float = 0.2, usage: dict = None
    ) -> tuple[str, list[dict]]:
        """Let the model call tools before it answers; returns (answer, the calls it made).

        `tools` are {"name", "description", "parameters" (a JSON schema)};
        `call_tool(name, args)` runs one and returns a JSON-able dict. The
        model gets MAX_TOOL_ROUNDS rounds of calls; the next call is made
        with tools switched off, so it has to answer in text.
        """
        declarations = [
            types.FunctionDeclaration(name=t["name"], description=t["description"], parameters_json_schema=t["parameters"])
            for t in tools
        ]
        with_tools = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            tools=[types.Tool(function_declarations=declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # we run the tools
        )
        text_only = with_tools.model_copy(
            update={"tool_config": types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE"))}
        )
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        calls = []
        for round_ in range(MAX_TOOL_ROUNDS + 1):
            response = self._complete(contents, with_tools if round_ < MAX_TOOL_ROUNDS else text_only, usage)
            requested = (response.function_calls or [])[:MAX_CALLS_PER_ROUND]
            if not requested:
                return (response.text or "").strip(), calls
            # the model's turn goes back as it came: thinking models sign it, and want the signature returned
            contents.append(response.candidates[0].content)
            parts = []
            for call in requested:
                args = dict(call.args or {})
                result = call_tool(call.name, args)
                calls.append({"name": call.name, "args": args, "result": result})
                parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=parts))
        return "", calls  # not reached: the last round has tools switched off

    def _complete(self, contents, config, usage: dict = None):
        """One model call, with retries for transient errors."""
        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            raise GenerationError("GEMINI_API_KEY is not set (see .env.example)")
        if usage is not None:
            usage["llm_calls"] = usage.get("llm_calls", 0) + 1
        api_failures = network_failures = 0
        while True:
            if self._client is None or api_failures or network_failures:
                self._client = _new_client()  # never retry on a connection pool that just failed
            try:
                response = self._client.models.generate_content(model=self.model, contents=contents, config=config)
                metadata = getattr(response, "usage_metadata", None)
                if usage is not None and metadata is not None:
                    usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + (metadata.prompt_token_count or 0)
                    usage["output_tokens"] = usage.get("output_tokens", 0) + (
                        (metadata.candidates_token_count or 0) + (getattr(metadata, "thoughts_token_count", 0) or 0)
                    )
                return response
            except errors.APIError as exc:
                message = f"Gemini API error ({exc.code}): {exc.message}"
                if exc.code not in RETRYABLE_STATUS_CODES:
                    raise GenerationError(message) from exc
                hinted = retry_delay_seconds(exc.message)
                if hinted > MAX_WAIT_SECONDS:  # e.g. the free tier's daily request quota is used up
                    raise GenerationError(
                        f"Gemini quota exhausted for {self.model}; it resets in about "
                        f"{hinted / 3600:.1f} h. Try again later or set GEMINI_MODEL to another model."
                    ) from exc
                api_failures += 1
                if api_failures >= MAX_RETRIES:
                    raise GenerationError(message) from exc
                # per-minute rate limits (429) reset in ~tens of seconds; 503 "high demand" is similar
                time.sleep(hinted or (20 if exc.code == 429 else min(2**api_failures, 15)))
            except httpx.TransportError as exc:
                # e.g. intermittent "SSL: INVALID_SESSION_ID" behind antivirus / TLS-inspecting software
                network_failures += 1
                if network_failures >= MAX_NETWORK_RETRIES:
                    raise GenerationError(f"Network error contacting Gemini: {exc}") from exc
                time.sleep(1)


_generator: GeminiGenerator = None


def get_generator() -> GeminiGenerator:
    """Process-wide generator (tests replace it via generator._generator)."""
    global _generator
    if _generator is None:
        _generator = GeminiGenerator()
    return _generator
