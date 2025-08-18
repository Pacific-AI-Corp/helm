import os
import pytest
import tempfile

from helm.common.cache import BlackHoleCacheConfig, SqliteCacheConfig
from helm.common.request import Request
from helm.clients.openrouter_client import OpenRouterClient


class TestOpenRouterClient:
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path: str = cache_file.name

    def teardown_method(self, method):
        os.remove(self.cache_path)

    @pytest.mark.parametrize(
        "model_name,test_input,expected_model",
        [
            (
                "openrouter/mistralai/mistral-medium-3.1",
                Request(
                    model="openrouter/mistralai/mistral-medium-3.1",
                    model_deployment="openrouter/mistralai/mistral-medium-3.1",
                ),
                "openrouter/mistralai/mistral-medium-3.1",
            ),
            (
                None,
                Request(model="openai/gpt-oss-20b:free", model_deployment="openrouter/openai/gpt-oss-20b:free"),
                "openai/gpt-oss-20b:free",
            ),
        ],
    )
    def test_get_model_for_request(self, model_name, test_input, expected_model):
        client = OpenRouterClient(
            tokenizer=None,
            tokenizer_name=None,
            cache_config=SqliteCacheConfig(self.cache_path),
            model_name=model_name,
            api_key="test_key",
        )
        assert client._get_model_for_request(test_input) == expected_model

    def test_api_key_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
        client = OpenRouterClient(
            tokenizer=None,
            tokenizer_name=None,
            cache_config=SqliteCacheConfig(self.cache_path),
        )
        assert client.api_key == "test_key"

    def test_api_key_argument(self):
        client = OpenRouterClient(
            tokenizer=None,
            tokenizer_name=None,
            cache_config=BlackHoleCacheConfig(),
            api_key="explicit_key",
        )
        assert client.api_key == "explicit_key"
