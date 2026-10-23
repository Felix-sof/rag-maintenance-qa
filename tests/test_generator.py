"""Tests for generator.py's retry policy, with a scripted fake Gemini client (no network)."""

import httpx
import pytest
from google.genai import errors

import generator
from generator import GeminiGenerator, GenerationError


class FakeResponse:
    text = " cevap "
    function_calls = None


def api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": f"error {code}"}})


@pytest.fixture
def script(monkeypatch):
    """Make every new client replay `outcomes` in order: an exception to raise, or a response."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    outcomes = []
    clients = []
    requests = []  # the keyword arguments of every generate_content call

    class FakeModels:
        def generate_content(self, **kwargs):
            requests.append(kwargs)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    class FakeClient:
        models = FakeModels()
        sent = requests  # shared by every client: what was sent, in order

    def new_client():
        clients.append(FakeClient())
        return clients[-1]

    monkeypatch.setattr(generator, "_new_client", new_client)
    monkeypatch.setattr(generator.time, "sleep", lambda s: None)
    return outcomes, clients


def test_dropped_connections_are_retried_on_fresh_clients(script):
    outcomes, clients = script
    outcomes += [httpx.ConnectError("SSL: INVALID_SESSION_ID")] * 3 + [FakeResponse()]
    assert GeminiGenerator().generate("sys", "prompt") == "cevap"
    assert len(clients) == 4  # a new client per attempt, never the pool that just failed


def test_gives_up_after_max_network_retries(script):
    outcomes, _ = script
    outcomes += [httpx.ConnectError("boom")] * generator.MAX_NETWORK_RETRIES
    with pytest.raises(GenerationError, match="Network error"):
        GeminiGenerator().generate("sys", "prompt")


def test_overloaded_model_is_retried(script):
    outcomes, _ = script
    outcomes += [api_error(503), api_error(429), FakeResponse()]
    assert GeminiGenerator().generate("sys", "prompt") == "cevap"


def test_gives_up_after_max_api_retries(script):
    outcomes, _ = script
    outcomes += [api_error(503)] * generator.MAX_RETRIES
    with pytest.raises(GenerationError, match="503"):
        GeminiGenerator().generate("sys", "prompt")


def test_non_retryable_error_fails_immediately(script):
    outcomes, clients = script
    outcomes += [api_error(400), FakeResponse()]
    with pytest.raises(GenerationError, match="400"):
        GeminiGenerator().generate("sys", "prompt")
    assert len(clients) == 1


def test_retry_delay_hint_is_parsed():
    assert generator.retry_delay_seconds("Please retry in 6h51m33.5s.") == pytest.approx(24693.5)
    assert generator.retry_delay_seconds("Please retry in 12.3s.") == pytest.approx(12.3)
    assert generator.retry_delay_seconds("no hint here") == 0


def test_exhausted_daily_quota_fails_fast_without_waiting(script):
    outcomes, clients = script
    outcomes += [api_error_with(429, "Quota exceeded. Please retry in 6h51m33s."), FakeResponse()]
    with pytest.raises(GenerationError, match="quota exhausted"):
        GeminiGenerator().generate("sys", "prompt")
    assert len(clients) == 1


def api_error_with(code, message):
    return errors.APIError(code, {"error": {"code": code, "message": message}})


def test_token_usage_is_added_to_the_callers_counter(script):
    class Usage:
        prompt_token_count, candidates_token_count, thoughts_token_count = 120, 30, 15

    class ResponseWithUsage(FakeResponse):
        usage_metadata = Usage()

    outcomes, _ = script
    outcomes += [ResponseWithUsage(), ResponseWithUsage()]
    usage = {}
    GeminiGenerator().generate("sys", "prompt", usage=usage)
    GeminiGenerator().generate("sys", "prompt", usage=usage)
    assert usage == {"llm_calls": 2, "prompt_tokens": 240, "output_tokens": 90}  # thought tokens count as output


def test_a_missing_api_key_is_a_generation_error_not_a_crash(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="GEMINI_API_KEY"):
        GeminiGenerator().generate("sys", "prompt")


def test_tool_calls_are_run_and_the_model_then_has_to_answer(script):
    from google.genai import types

    class AsksForATool:
        text = None
        function_calls = [types.FunctionCall(name="assess_process", args={"torque_nm": 45})]
        candidates = [types.Candidate(content=types.Content(role="model", parts=[types.Part(text="calling")]))]

    outcomes, clients = script
    outcomes += [AsksForATool(), FakeResponse()]
    ran = []
    tools = [{"name": "assess_process", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    answer, calls = GeminiGenerator().generate_with_tools(
        "sys", "prompt", tools, lambda name, args: ran.append((name, args)) or {"ok": True}
    )

    assert answer == "cevap"
    assert ran == [("assess_process", {"torque_nm": 45})]
    assert calls == [{"name": "assess_process", "args": {"torque_nm": 45}, "result": {"ok": True}}]
    first, second = clients[0].sent
    assert first["config"].tool_config is None  # tools on
    assert second["config"].tool_config.function_calling_config.mode == "NONE"  # the last round must answer
    # the second request carries the question, the model's tool request and the tool's result
    assert [c.role for c in second["contents"]] == ["user", "model", "user"]
    assert second["contents"][2].parts[0].function_response.response == {"result": {"ok": True}}
