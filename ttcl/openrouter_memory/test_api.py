import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from ttcl.common.openrouter import APIStop, Ledger, OpenRouter, exact_model
from ttcl.openrouter_memory.launch import read_key


INFO = {
    "id": "openai/gpt-6-astra",
    "context_length": 1050000,
    "pricing": {"prompt": "0.00001", "completion": "0.00005"},
    "supported_parameters": ["seed", "reasoning", "max_tokens"],
    "top_provider": {"max_completion_tokens": 128000},
}


def reply(text='{"ok":true}', **extra):
    return io.BytesIO(
        json.dumps(
            {
                "model": INFO["id"],
                "provider": "test-provider",
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
                **extra,
            }
        ).encode()
    )


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = Ledger(self.root, 50)
        self.messages = [{"role": "user", "content": "Return JSON"}]

    def backend(self, opener):
        return OpenRouter(
            "FAKE_SECRET_NOT_FOR_LOGS",
            INFO,
            None,
            self.ledger,
            opener=opener,
            sleep=lambda _: None,
        )

    def test_exact_model_no_silent_substitution(self):
        with self.assertRaises(ValueError):
            exact_model([INFO], "openai/not-available")

    def test_supported_parameters_and_usage(self):
        calls = []

        def opener(request, **kwargs):
            calls.append(request)
            return reply()

        result = self.backend(opener).generate(self.messages, 2**60)
        body = json.loads(calls[0].data)
        self.assertEqual(
            calls[0].get_header("Authorization"), "Bearer FAKE_SECRET_NOT_FOR_LOGS"
        )
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)
        self.assertFalse(body["provider"]["allow_fallbacks"])
        self.assertEqual(body["reasoning"], {"effort": "medium", "exclude": True})
        self.assertLess(body["seed"], 2**31)
        self.assertEqual(result["input_tokens"], 10)
        self.assertEqual(result["output_tokens"], 5)
        self.assertAlmostEqual(self.ledger.charged, 0.001)
        self.assertNotIn(
            "FAKE_SECRET_NOT_FOR_LOGS",
            "".join(p.read_text() for p in self.root.iterdir()),
        )

    def test_budget_stops_before_http(self):
        self.ledger.limit = 0.0001
        with self.assertRaises(APIStop):
            self.backend(lambda *a, **k: self.fail("should not contact API")).generate(
                self.messages, 0
            )
        self.assertEqual(self.ledger.requests, 0)

    def test_429_retries_and_counts_attempts(self):
        attempts = []

        def opener(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise urllib.error.HTTPError(
                    "https://openrouter.ai",
                    429,
                    "contains FAKE_SECRET_NOT_FOR_LOGS",
                    {},
                    None,
                )
            return reply()

        self.backend(opener).generate(self.messages, 0)
        self.assertEqual(self.ledger.requests, 2)
        self.assertAlmostEqual(self.ledger.charged, 0.001)
        self.assertNotIn(
            "FAKE_SECRET_NOT_FOR_LOGS", (self.root / "api_calls.jsonl").read_text()
        )

    def test_auth_failure_stops_without_retry_or_secret(self):
        def opener(*args, **kwargs):
            raise urllib.error.HTTPError(
                "https://openrouter.ai", 401, "FAKE_SECRET_NOT_FOR_LOGS", {}, None
            )

        with self.assertRaises(APIStop) as caught:
            self.backend(opener).generate(self.messages, 0)
        self.assertEqual(self.ledger.requests, 1)
        self.assertNotIn("FAKE_SECRET_NOT_FOR_LOGS", str(caught.exception))
        self.assertIsNotNone(self.ledger.stop_reason)

    def test_timeout_unknown_cost_is_reserved(self):
        def opener(*args, **kwargs):
            raise TimeoutError("FAKE_SECRET_NOT_FOR_LOGS")

        with self.assertRaises(APIStop):
            self.backend(opener).generate(self.messages, 0)
        self.assertEqual(self.ledger.requests, 3)
        self.assertEqual(self.ledger.unknown_cost_calls, 3)
        self.assertGreater(self.ledger.charged, 0)

    def test_missing_usage_never_fabricates_tokens(self):
        with self.assertRaises(APIStop):
            self.backend(lambda *a, **k: reply(usage={})).generate(self.messages, 0)
        self.assertEqual(self.ledger.unknown_cost_calls, 1)

    def test_non_tty_never_reads_echoed_key(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.stdin.isatty", return_value=False),
            patch("getpass.getpass") as prompt,
        ):
            with self.assertRaises(ValueError):
                read_key()
            prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
