"""CPU-only OpenRouter chat backend with explicit model and spending records.

The API key exists only in memory/environment and HTTP Authorization headers.
No request headers or environment snapshots are written to experiment files.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

API = "https://openrouter.ai/api/v1"


class APIStop(RuntimeError):
    """Stop the experiment, rather than treating infrastructure as task failure."""


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def catalog():
    request = urllib.request.Request(
        API + "/models", headers={"User-Agent": "CLBench-memory-experiment"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["data"]


def exact_model(models, identity):
    for model in models:
        if model["id"] == identity:
            return model
    raise ValueError(
        f"OpenRouter 模型目录没有精确 ID {identity!r}；未自动替换模型。用 --list-models 查看。"
    )


class BankTokenizer:
    """Use the ORIGINAL Qwen tokenizer on CPU solely to keep bank budgets equal.

    It is not used as the claimed GPT token count. Actual API usage is recorded
    separately; provider context preflight uses a conservative byte estimate.
    """

    def __init__(self, path):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))

    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


class Ledger:
    def __init__(self, root, max_cost_usd=50.0, max_requests=1500):
        if not math.isfinite(max_cost_usd) or max_cost_usd <= 0 or max_requests <= 0:
            raise ValueError("API cost/request limits must be positive")
        self.root = Path(root)
        self.limit = max_cost_usd
        self.max_requests = max_requests
        self.charged = 0.0
        self.requests = 0
        self.known_cost = 0.0
        self.unknown_cost_calls = 0
        self.stop_reason = None

    def reserve(self, amount):
        if self.requests >= self.max_requests or self.charged + amount > self.limit:
            self.stop_reason = (
                "API budget reached before next request (conservative reservation)."
            )
            raise APIStop(self.stop_reason)
        self.requests += 1
        # Persist the reservation before submitting a possibly billable request.
        self.charged += amount
        self.persist()

    def settle(self, reserved, actual, event):
        if actual is None:
            self.unknown_cost_calls += 1
            accounted = reserved
        else:
            accounted = actual
            self.known_cost += actual
            self.charged += actual - reserved
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "api_calls.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        **event,
                        "reported_cost_usd": actual,
                        "accounted_cost_usd": accounted,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self.persist()

    def persist(self):
        write(
            self.root / "api_usage.json",
            {
                "http_requests": self.requests,
                "reported_cost_usd": self.known_cost,
                "accounted_cost_usd": self.charged,
                "unknown_cost_calls": self.unknown_cost_calls,
                "max_cost_usd": self.limit,
                "max_requests": self.max_requests,
                "cost_note": "Unknown costs and in-flight calls use conservative reservations, not a claim of actual billing.",
            },
        )


def prices(info):
    pricing = info.get("pricing", {})
    # Include long-context price tiers in the reservation rather than silently
    # underestimating a call that crosses a provider's pricing threshold.
    prompt = max(
        float(p.get("prompt", pricing.get("prompt", 0)))
        for p in [pricing, *pricing.get("overrides", [])]
    )
    completion = max(
        float(p.get("completion", pricing.get("completion", 0)))
        for p in [pricing, *pricing.get("overrides", [])]
    )
    request = float(pricing.get("request", 0))
    if any(not math.isfinite(v) or v < 0 for v in [prompt, completion, request]):
        raise ValueError("Invalid catalog pricing")
    return prompt, completion, request


class OpenRouter:
    def __init__(
        self,
        key,
        info,
        tokenizer,
        ledger,
        *,
        output_tokens=8192,
        reasoning="medium",
        provider=None,
        timeout=180,
        retries=2,
        opener=None,
        sleep=time.sleep,
    ):
        if not key or not key.strip():
            raise ValueError("OPENROUTER_API_KEY is missing")
        self._key = key.strip()
        self.info = info
        self.model_id = info["id"]
        self.tokenizer = tokenizer
        self.ledger = ledger
        self.output_tokens = output_tokens
        self.reasoning = reasoning
        self.provider = provider
        self.timeout, self.retries = timeout, retries
        self.opener = opener or urllib.request.urlopen
        self.sleep = sleep
        self.purpose = {}
        # Fail configuration checks before any potentially billable actor call.
        self.payload([], 0, output_tokens, 0.7)
        prices(info)

    def payload(self, messages, seed, limit, temperature):
        supported = set(self.info.get("supported_parameters", []))
        body = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "max_tokens": limit,
            "transforms": [],
            "provider": {"allow_fallbacks": False, "require_parameters": True},
        }
        if self.provider:
            body["provider"]["only"] = [self.provider]
        omitted = []
        for name, value in [
            ("seed", seed % (2**31)),
            ("temperature", temperature),
            ("top_p", 0.9),
        ]:
            if name in supported:
                body[name] = value
            else:
                omitted.append(name)
        if self.reasoning != "default":
            if "reasoning" not in supported:
                raise ValueError(
                    "This model does not advertise reasoning control; choose --reasoning default"
                )
            body["reasoning"] = {"effort": self.reasoning, "exclude": True}
            efforts = (self.info.get("reasoning") or {}).get("supported_efforts")
            if efforts and self.reasoning not in efforts:
                raise ValueError(
                    f"Reasoning effort {self.reasoning!r} is not supported by this catalog model"
                )
        return body, omitted

    def generate(self, messages, seed, *, max_new_tokens=None, temperature=0.7):
        limit = max_new_tokens or self.output_tokens
        body, omitted = self.payload(messages, seed, limit, temperature)
        serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        estimated_upper_input = (
            len(serialized.encode("utf-8")) + 1024 + 32 * len(messages)
        )
        context = self.info.get("context_length", 0)
        if estimated_upper_input + limit > context:
            raise ValueError(
                "Conservative input/output context check failed; no history was truncated"
            )
        provider_max = self.info.get("top_provider", {}).get("max_completion_tokens")
        if provider_max and limit > provider_max:
            raise ValueError("Output limit exceeds catalog provider limit")
        p_in, p_out, p_request = prices(self.info)
        reserve = estimated_upper_input * p_in + limit * p_out + p_request
        prompt_hash = hashlib.sha256(serialized.encode()).hexdigest()
        base_event = {
            **self.purpose,
            "requested_model": self.model_id,
            "prompt_sha256": prompt_hash,
            "requested_seed": seed,
            "effective_seed": body.get("seed"),
            "omitted_parameters": omitted,
        }
        for attempt in range(self.retries + 1):
            self.ledger.reserve(reserve)
            start = time.monotonic()
            request = urllib.request.Request(
                API + "/chat/completions",
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": "Bearer " + self._key,
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "CLBench autonomous experience bank",
                    "X-OpenRouter-Cache": "false",
                },
                method="POST",
            )
            response = None
            try:
                with self.opener(request, timeout=self.timeout) as stream:
                    response = json.load(stream)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                code = getattr(exc, "code", None)
                # Never log HTTP response bodies or exception strings: upstream
                # errors may echo request headers/credentials.
                event = {
                    **base_event,
                    "attempt": attempt,
                    "status": "http_error",
                    "http_status": code,
                    "error_type": type(exc).__name__,
                    "seconds": time.monotonic() - start,
                }
                self.ledger.settle(
                    reserve,
                    0.0 if code in {400, 401, 402, 403, 404, 422, 429} else None,
                    event,
                )
                if (
                    code in {429, 500, 502, 503, 504, 529} or code is None
                ) and attempt < self.retries:
                    self.sleep(min(2 ** (attempt + 1), 8))
                    continue
                self.ledger.stop_reason = f"OpenRouter request failed ({type(exc).__name__}, HTTP {code}); inspect api_calls.jsonl."
                raise APIStop(self.ledger.stop_reason) from None
            except (ValueError, UnicodeError):
                self.ledger.settle(
                    reserve,
                    None,
                    {**base_event, "status": "invalid_json", "attempt": attempt},
                )
                self.ledger.stop_reason = "OpenRouter returned invalid JSON."
                raise APIStop(self.ledger.stop_reason) from None
            usage = response.get("usage") or {}
            actual = usage.get("cost")
            if (
                not isinstance(actual, (int, float))
                or not math.isfinite(actual)
                or actual < 0
            ):
                actual = None
            error = response.get("error")
            event = {
                **base_event,
                "attempt": attempt,
                "status": "api_error" if error else "response",
                "id": response.get("id"),
                "resolved_model": response.get("model"),
                "provider": response.get("provider"),
                "usage": usage,
                "seconds": time.monotonic() - start,
            }
            # usage is provider metadata, but still remove any accidentally echoed key.
            event = json.loads(json.dumps(event).replace(self._key, "[REDACTED]"))
            self.ledger.settle(reserve, actual, event)
            if error:
                self.ledger.stop_reason = (
                    "OpenRouter returned an API error; response body was not logged."
                )
                raise APIStop(self.ledger.stop_reason)
            try:
                choice = response["choices"][0]
                content = choice["message"].get("content")
                if content is None:
                    content = ""  # e.g. all completion budget consumed by reasoning
                if not isinstance(content, str):
                    raise ValueError("Non-text content")
                input_tokens, output_tokens = (
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                )
                if any(
                    type(v) is not int or v < 0 for v in [input_tokens, output_tokens]
                ):
                    raise ValueError("Invalid usage")
            except (KeyError, IndexError, TypeError, ValueError):
                self.ledger.stop_reason = (
                    "OpenRouter response missing text/choice or valid usage."
                )
                raise APIStop(self.ledger.stop_reason) from None
            return {
                "raw_response": content.strip(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "context_limit": context,
                "finish_reason": choice.get("finish_reason"),
                "rendered_prompt_sha256": prompt_hash,
                "prompt_hash_format": "canonical_messages_json",
                "api_model": response.get("model"),
                "requested_model": self.model_id,
                "api_provider": response.get("provider"),
                "api_response_id": response.get("id"),
                "system_fingerprint": response.get("system_fingerprint"),
                "api_usage": usage,
                "reported_cost_usd": actual,
                "request_parameters": {
                    k: v for k, v in body.items() if k != "messages"
                },
                "omitted_parameters": omitted,
            }
        raise AssertionError("Unreachable retry state")
