import pytest
from ai_handler.ai_handler import AiHandler
from ai_handler.question import SimpleQuestion, Question
from ai_handler.answer import SimpleAnswer, Answer
from ai_handler.cache import InMemoryCache, NullCache
from ai_handler.providers.ai_provider_client import AiProviderClient
from ai_handler.errors import InvalidModelResponseException


class DummyProvider(AiProviderClient):
    def ask(self, prompt: str | Question, **kwargs) -> str:
        if isinstance(prompt, Question):
            if "bad" in prompt.question:
                return ""  # Simulate unparseable response
            else:
                return prompt.question.upper()
        if isinstance(prompt, str):
            if "bad" in prompt:
                return ""  # Simulate unparseable response
            else:
                return prompt.upper()


def test_handler_ask_success(monkeypatch):
    handler = AiHandler(DummyProvider(), InMemoryCache())
    answer = handler.ask("hi there")
    assert isinstance(answer, SimpleAnswer)
    assert answer.raw == "HI THERE"


def test_handler_cache_hit():
    cache = InMemoryCache()
    q = SimpleQuestion("hi cache")
    cache.set(q, "CACHED ANSWER")
    handler = AiHandler(DummyProvider(), cache)
    answer = handler.ask(q)
    assert answer.raw == "CACHED ANSWER"


def test_handler_cache_miss(monkeypatch):
    handler = AiHandler(DummyProvider(), InMemoryCache())
    answer = handler.ask("not cached")
    assert answer.raw == "NOT CACHED"


def test_handler_ask_with_question_object():
    handler = AiHandler(DummyProvider(), InMemoryCache())
    q = SimpleQuestion("what is up?")
    answer = handler.ask(q)
    assert answer.raw == "WHAT IS UP?"


def test_handler_null_cache():
    handler = AiHandler(DummyProvider(), NullCache())
    answer = handler.ask("always miss cache")
    assert answer.raw == "ALWAYS MISS CACHE"


def test_handler_invalid_response_and_retry():
    class BadProvider(AiProviderClient):
        def ask(self, prompt: str, **kwargs) -> str:
            return ""  # Always fails parsing

    handler = AiHandler(BadProvider(), NullCache())
    q = SimpleQuestion("bad input")
    with pytest.raises(InvalidModelResponseException):
        handler.ask(q)


def test_handler_custom_answer_type():
    # Custom answer class that reverses the string
    class ReverseAnswer(Answer):
        def __init__(self, raw: str):
            super().__init__(raw)
            self.reversed = raw[::-1]

        @classmethod
        def parse(cls, raw: str):
            if not raw:
                raise ValueError("Empty response")
            return cls(raw)

    class ReverseProvider(AiProviderClient):
        def ask(self, prompt: str, **kwargs) -> str:
            return "reverse me"

    handler = AiHandler(ReverseProvider(), NullCache())
    answer = handler.ask("foo", response_factory=ReverseAnswer)
    assert isinstance(answer, ReverseAnswer)
    assert answer.reversed == "em esrever"


def test_handler_max_retries_exceeded(monkeypatch):
    # Provider always returns bad response, max_retries=1 should raise after 1 try
    class AlwaysBad(AiProviderClient):
        def ask(self, prompt: str, **kwargs) -> str:
            return ""

    q = SimpleQuestion("bad for retry test")
    handler = AiHandler(AlwaysBad(), NullCache())
    with pytest.raises(InvalidModelResponseException):
        handler.ask(q)
