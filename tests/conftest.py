import pytest
from ai_handler.ai_handler import AiHandler
from ai_handler.answer import Answer
from ai_handler.question import Question
from ai_handler.cache import InMemoryCache, NullCache
from ai_handler.errors import InvalidModelResponseException
from ai_handler.ai_provider_client import AiProviderClient

class DummyProvider(AiProviderClient):
    def ask(self, prompt: str | Question, **kwargs) -> str:
        if isinstance(prompt, Question):
            if "error" in prompt.question:
                # Simulate model returning bad format for error case
                return "NOT_A_VALID_ANSWER"
            return prompt.question.upper()
        if isinstance(prompt, str):
            if "error" in prompt:
                # Simulate model returning bad format for error case
                return "NOT_A_VALID_ANSWER"
            return prompt.upper()

@pytest.fixture
def dummy_provider():
    return DummyProvider()

@pytest.fixture
def in_memory_cache():
    return InMemoryCache()

@pytest.fixture
def null_cache():
    return NullCache()

@pytest.fixture
def handler(dummy_provider, in_memory_cache):
    return AiHandler(client=dummy_provider, cache=in_memory_cache)

