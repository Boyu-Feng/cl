from types import SimpleNamespace
import unittest

from src.vendors.ace.llm import timed_llm_call


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class ACELLMTests(unittest.TestCase):
    def test_openai_omits_default_temperature(self):
        client = _FakeClient()

        response, call_info = timed_llm_call(
            client=client,
            api_provider="openai",
            model="gpt-5",
            prompt="Hello",
            role="generator",
            call_id="test_call",
            use_json_mode=True,
        )

        api_params = client.chat.completions.calls[0]
        self.assertNotIn("temperature", api_params)
        self.assertEqual(api_params["max_completion_tokens"], 4096)
        self.assertEqual(api_params["response_format"], {"type": "json_object"})
        self.assertEqual(response, '{"ok": true}')
        self.assertGreater(call_info["cost_usd"], 0.0)
        self.assertTrue(call_info["pricing_complete"])

    def test_openai_preserves_explicit_temperature_override(self):
        client = _FakeClient()

        timed_llm_call(
            client=client,
            api_provider="openai",
            model="gpt-4o",
            prompt="Hello",
            role="generator",
            call_id="test_call",
            extra_api_params={"temperature": 0.0},
        )

        api_params = client.chat.completions.calls[0]
        self.assertEqual(api_params["temperature"], 0.0)

    def test_non_openai_keeps_zero_temperature_default(self):
        client = _FakeClient()

        timed_llm_call(
            client=client,
            api_provider="together",
            model="some-model",
            prompt="Hello",
            role="generator",
            call_id="test_call",
        )

        api_params = client.chat.completions.calls[0]
        self.assertEqual(api_params["temperature"], 0.0)
        self.assertEqual(api_params["max_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()
