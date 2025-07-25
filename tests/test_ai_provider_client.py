from ai_handler.providers.ai_provider_client import AiProviderClient


class DummyProvider(AiProviderClient):
    def ask(self, prompt: str, **kwargs) -> str:
        return prompt[::-1]  # reverses input


def test_ask_returns_expected_result():
    provider = DummyProvider()
    assert provider.ask("abc") == "cba"
