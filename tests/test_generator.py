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


def test_gives_up_after_max_rate_limit_retries(script):
    outcomes, _ = script
    outcomes += [api_error(429)] * generator.MAX_RETRIES
    with pytest.raises(GenerationError, match="429") as failure:
        GeminiGenerator().generate("sys", "prompt")
    assert not failure.value.switch_model  # a per-minute limit isn't the model's fault


def test_an_overloaded_model_gives_up_after_a_few_tries(script):
    outcomes, _ = script
    outcomes += [api_error(503)] * (generator.MAX_OVERLOAD_RETRIES + 1)
    with pytest.raises(GenerationError, match="503") as failure:
        GeminiGenerator(fallback=None).generate("sys", "prompt")
    assert failure.value.switch_model


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
        GeminiGenerator(fallback=None).generate("sys", "prompt")
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
    assert usage == {"llm_calls": 2, "model": generator.MODEL, "prompt_tokens": 240, "output_tokens": 90}  # thoughts count


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


def test_a_stalled_request_times_out_instead_of_hanging(script):
    outcomes, clients = script
    outcomes += [httpx.ReadTimeout("no answer")] * 2
    with pytest.raises(GenerationError, match="didn't answer"):
        GeminiGenerator(fallback=None).generate("sys", "prompt")
    assert len(clients) == 2  # retried once, on a fresh connection


def test_one_stall_is_retried(script):
    outcomes, _ = script
    outcomes += [httpx.ReadTimeout("no answer"), FakeResponse()]
    assert GeminiGenerator().generate("sys", "prompt") == "cevap"


def test_the_client_has_a_timeout_and_connects_over_ipv4(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    transports = []
    real = httpx.HTTPTransport
    monkeypatch.setattr(httpx, "HTTPTransport", lambda **kwargs: transports.append(kwargs) or real(**kwargs))
    client = generator._new_client()
    assert client._api_client._http_options.timeout == generator.TIMEOUT_SECONDS * 1000
    assert transports[0]["local_address"] == "0.0.0.0"


class TestFallbackModel:
    """When the main model is overloaded, stalls or is out of quota, the fallback model answers."""

    def test_overload_switches_to_the_fallback_at_once(self, script):
        outcomes, clients = script
        outcomes += [api_error(503), FakeResponse()]
        usage = {}
        assert GeminiGenerator("main", fallback="spare").generate("sys", "prompt", usage=usage) == "cevap"
        assert [r["model"] for r in clients[0].sent] == ["main", "spare"]
        assert usage["model"] == "spare" and usage["llm_calls"] == 2

    def test_a_stall_switches_too_and_the_fallback_gets_the_retries(self, script):
        outcomes, clients = script
        outcomes += [httpx.ReadTimeout("stalled"), api_error(503), FakeResponse()]
        assert GeminiGenerator("main", fallback="spare").generate("sys", "prompt") == "cevap"
        assert [r["model"] for r in clients[0].sent] == ["main", "spare", "spare"]

    def test_used_up_quota_switches_at_once(self, script):
        outcomes, clients = script
        outcomes += [api_error_with(429, "Quota exceeded. Please retry in 6h51m33s."), FakeResponse()]
        assert GeminiGenerator("main", fallback="spare").generate("sys", "prompt") == "cevap"
        assert [r["model"] for r in clients[0].sent] == ["main", "spare"]

    def test_errors_any_model_would_hit_dont_switch(self, script):
        outcomes, clients = script
        outcomes += [api_error(400), FakeResponse()]
        with pytest.raises(GenerationError, match="400"):
            GeminiGenerator("main", fallback="spare").generate("sys", "prompt")
        assert [r["model"] for r in clients[0].sent] == ["main"]

    def test_a_tool_conversation_stays_on_the_model_it_started_on(self, script):
        from google.genai import types

        class AsksForATool:
            text = None
            function_calls = [types.FunctionCall(name="t", args={})]
            candidates = [types.Candidate(content=types.Content(role="model", parts=[types.Part(text="calling")]))]

        outcomes, clients = script
        outcomes += [api_error(503), AsksForATool(), FakeResponse()]
        tools = [{"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}]
        GeminiGenerator("main", fallback="spare").generate_with_tools("sys", "p", tools, lambda n, a: {})
        # the fallback took the first round, so it gets the second: its signed turn goes back to it
        assert [r["model"] for r in clients[0].sent][-2:] == ["spare", "spare"]
