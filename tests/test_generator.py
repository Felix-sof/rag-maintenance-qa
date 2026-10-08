"""Tests for generator.py's retry policy, with a scripted fake Gemini client (no network)."""

import httpx
import pytest
from google.genai import errors

import generator
from generator import GeminiGenerator, GenerationError


class FakeResponse:
    text = " cevap "


def api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": f"error {code}"}})


@pytest.fixture
def script(monkeypatch):
    """Make every new client replay `outcomes` in order: an exception to raise, or a response."""
    outcomes = []
    clients = []

    class FakeModels:
        def generate_content(self, **kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    class FakeClient:
        models = FakeModels()

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
