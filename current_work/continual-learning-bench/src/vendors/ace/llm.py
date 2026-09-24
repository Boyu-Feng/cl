"""
==============================================================================
llm.py
==============================================================================

This file contains the LLM class for the project.

"""

from datetime import datetime
import random
import time

import openai

from ...usage import build_usage_event
from .logger import log_llm_call, log_problematic_request


def _build_api_params(
    api_provider,
    model,
    prompt,
    max_tokens,
    use_json_mode,
    extra_api_params,
):
    if api_provider == "openai":
        max_tokens_key = "max_completion_tokens"
    else:
        max_tokens_key = "max_tokens"

    api_params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        max_tokens_key: max_tokens,
    }

    # Newer OpenAI reasoning/chat models may reject any explicit temperature
    # override and require provider defaults. Preserve the old deterministic
    # default for non-OpenAI providers.
    if api_provider != "openai":
        api_params["temperature"] = 0.0

    if extra_api_params:
        api_params.update(extra_api_params)

    if use_json_mode:
        api_params["response_format"] = {"type": "json_object"}

    return api_params


def _build_usage_metadata(
    api_provider,
    model,
    prompt_tokens,
    response_tokens,
):
    event = build_usage_event(
        model=model,
        provider=api_provider,
        input_tokens=prompt_tokens,
        output_tokens=response_tokens,
        call_type="completion",
    )
    return {
        "provider": api_provider,
        "prompt_num_tokens": event.input_tokens,
        "response_num_tokens": event.output_tokens,
        "total_num_tokens": event.total_tokens,
        "cost_usd": event.cost_usd,
        "pricing_complete": event.cost_usd is not None,
        "pricing_source": event.pricing_source,
        "pricing_error": event.pricing_error,
    }


def timed_llm_call(
    client,
    api_provider,
    model,
    prompt,
    role,
    call_id,
    max_tokens=4096,
    log_dir=None,
    sleep_seconds=15,
    retries_on_timeout=1000,
    attempt=1,
    use_json_mode=False,
    extra_api_params=None,
):
    """
    Make a timed LLM call with error handling and retry logic.

    EMPTY RESPONSE HANDLING STRATEGY:
    - Training calls (call_id starts with 'train_'): Skip the entire training sample
    - Test calls (call_id starts with 'test_'): Mark as incorrect (return wrong answers)
    - All empty responses are logged to problematic_requests/ for SambaNova support analysis

    For test calls specifically: Returns "INCORRECT_DUE_TO_EMPTY_RESPONSE" repeated 4 times
    (comma-separated) to handle the 4-question format used in financial NER evaluation.

    Args:
        client: API client
        model: Model name to use
        prompt: Text prompt to send
        role: Role for logging (generator, reflector, curator)
        call_id: Unique identifier for this call (format: {train|test}_{role}_{details})
        max_tokens: Maximum tokens to generate
        log_dir: Directory for detailed logging
        sleep_seconds: Base sleep time between retries
        retries_on_timeout: Maximum number of retries for timeouts/rate limits/empty responses
        attempt: Current attempt number (for recursive calls)
        use_json_mode: Whether to use JSON mode for structured output

    Returns:
        tuple: (response_text, call_info_dict)

    Special return values for empty responses:
        - Training: ("INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, ...", call_info)
        - Testing: ("INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, ...", call_info)
    """
    start_time = time.time()
    prompt_time = time.time()

    print(f"[{role.upper()}] Starting call {call_id}...")

    # Check if we're using API key mixer for dynamic key rotation on retries
    using_key_mixer = False

    while True:
        try:
            # Get client
            active_client = client

            # Prepare API call parameters
            api_params = _build_api_params(
                api_provider=api_provider,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                use_json_mode=use_json_mode,
                extra_api_params=extra_api_params,
            )
            call_start = time.time()
            response = active_client.chat.completions.create(**api_params)
            call_end = time.time()

            # Check if response is valid
            if not response or not response.choices or len(response.choices) == 0:
                raise Exception("Empty response from API")

            response_time = time.time()
            total_time = response_time - start_time
            response_content = response.choices[0].message.content

            if response_content is None:
                raise Exception("API returned None content")

            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "response": response_content,
                "prompt_time": prompt_time - start_time,
                "response_time": response_time - prompt_time,
                "total_time": total_time,
                "call_time": call_end - call_start,
                "prompt_length": len(prompt),
                "response_length": len(response_content),
            }
            call_info.update(
                _build_usage_metadata(
                    api_provider=api_provider,
                    model=model,
                    prompt_tokens=response.usage.prompt_tokens,
                    response_tokens=response.usage.completion_tokens,
                )
            )

            print(f"[{role.upper()}] Call {call_id} completed in {total_time:.2f}s")

            if log_dir:
                log_llm_call(log_dir, call_info)

            return response_content, call_info

        except Exception as e:
            # Check for both timeout and rate limit errors
            is_timeout = any(
                k in str(e).lower() for k in ["timeout", "timed out", "connection"]
            )
            is_rate_limit = any(
                k in str(e).lower()
                for k in ["rate limit", "429", "rate_limit_exceeded"]
            )
            is_empty_response = (
                "empty response" in str(e).lower()
                or "api returned none content" in str(e).lower()
            )

            # Check for server errors (500, 502, 503, etc.) that should be retried
            is_server_error = False
            if hasattr(e, "response"):
                try:
                    status_code = getattr(e.response, "status_code", None)
                    if status_code and status_code >= 500:
                        is_server_error = True
                        print(
                            f"[{role.upper()}] Server error detected: HTTP {status_code}"
                        )
                except Exception:
                    pass

            # Also check for 500 errors in the error message itself
            if any(
                k in str(e).lower()
                for k in [
                    "500 internal server error",
                    "internal server error",
                    "502 bad gateway",
                    "503 service unavailable",
                ]
            ):
                is_server_error = True
                print(
                    f"[{role.upper()}] Server error detected in message: {str(e)[:100]}..."
                )

            # Also check for specific OpenAI exceptions
            if hasattr(openai, "RateLimitError") and isinstance(
                e, openai.RateLimitError
            ):
                is_rate_limit = True

            # Check for OpenAI InternalServerError
            if hasattr(openai, "InternalServerError") and isinstance(
                e, openai.InternalServerError
            ):
                is_server_error = True
                print(f"[{role.upper()}] OpenAI InternalServerError detected")

            # Debug empty response issues
            if is_empty_response:
                print(f"\n🚨 DEBUG: Empty response detected for {call_id}")
                print(f"📝 Exception type: {type(e).__name__}")
                print(f"📝 Exception message: {str(e)}")
                print(f"📝 Using JSON mode: {use_json_mode}")
                print(f"📝 Model: {model}")
                print(f"📝 Prompt length: {len(prompt)}")
                print("📝 Prompt preview (first 500 chars):")
                print(f"    {prompt[:500]}...")
                print(f"📝 Full exception details: {repr(e)}")
                if hasattr(e, "response"):
                    print(f"📝 Raw response object: {e.response}")
                    if hasattr(e.response, "text"):
                        print(f"📝 Raw response text: {e.response.text}")
                    if hasattr(e.response, "content"):
                        print(f"📝 Raw response content: {e.response.content}")
                print("-" * 60)

                # Log problematic requests for SambaNova support
                log_problematic_request(
                    call_id,
                    prompt,
                    model,
                    api_params,
                    e,
                    log_dir,
                    using_key_mixer,
                    client if using_key_mixer else None,
                )

            # For empty responses, we handle differently based on context
            if is_empty_response:
                # Log the problematic request for SambaNova support
                log_problematic_request(
                    call_id,
                    prompt,
                    model,
                    api_params,
                    e,
                    log_dir,
                    using_key_mixer,
                    client if using_key_mixer else None,
                )

                # Check if this is a training or test call to decide behavior
                if call_id.startswith("train_"):
                    # In training: Mark as incorrect answer (same as testing)
                    print(
                        f"[{role.upper()}] 🚨 Empty response in training - marking as INCORRECT for {call_id}"
                    )
                    error_time = time.time()
                    call_info = {
                        "role": role,
                        "call_id": call_id,
                        "model": model,
                        "prompt": prompt,
                        "error": "TRAINING_INCORRECT: " + str(e),
                        "total_time": error_time - start_time,
                        "prompt_length": len(prompt),
                        "response_length": 0,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                        "datetime": datetime.now().isoformat(),
                        "training_marked_incorrect_due_to_empty_response": True,
                    }
                    if log_dir:
                        log_llm_call(log_dir, call_info)

                    # Return a response that will be marked as incorrect
                    # For the 4-question format, we return 4 wrong answers
                    incorrect_response = "INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE"
                    return incorrect_response, call_info

                elif call_id.startswith("test_"):
                    # In testing: Treat as incorrect answer
                    print(
                        f"[{role.upper()}] 🚨 Empty response in testing - marking as INCORRECT for {call_id}"
                    )
                    error_time = time.time()
                    call_info = {
                        "role": role,
                        "call_id": call_id,
                        "model": model,
                        "prompt": prompt,
                        "error": "TEST_INCORRECT: " + str(e),
                        "total_time": error_time - start_time,
                        "prompt_length": len(prompt),
                        "response_length": 0,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                        "datetime": datetime.now().isoformat(),
                        "test_marked_incorrect_due_to_empty_response": True,
                    }
                    if log_dir:
                        log_llm_call(log_dir, call_info)

                    # Return a response that will be marked as incorrect
                    # For the 4-question format, we return 4 wrong answers
                    incorrect_response = "INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE"
                    return incorrect_response, call_info

            # Retry logic for timeouts, rate limits, and server errors
            if (
                is_timeout or is_rate_limit or is_server_error
            ) and attempt < retries_on_timeout:
                attempt += 1
                if is_rate_limit:
                    error_type = "rate limited"
                    base_sleep = sleep_seconds * 2
                elif is_server_error:
                    error_type = "server error (500+)"
                    base_sleep = sleep_seconds * 1.5  # Moderate delay for server errors
                elif is_empty_response:
                    error_type = "returned empty response"
                    base_sleep = sleep_seconds
                else:
                    error_type = "timed out"
                    base_sleep = sleep_seconds
                jitter = random.uniform(0.5, 1.5)  # Add jitter to avoid thundering herd
                sleep_time = base_sleep * jitter
                print(
                    f"[{role.upper()}] Call {call_id} {error_type}, sleeping {sleep_time:.1f}s then retrying "
                    f"({attempt}/{retries_on_timeout})..."
                )
                time.sleep(sleep_time)
                continue

            error_time = time.time()
            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "error": str(e),
                "total_time": error_time - start_time,
                "prompt_length": len(prompt),
                "attempt": attempt,
            }

            print(
                f"[{role.upper()}] Call {call_id} failed after {error_time - start_time:.2f}s: {e}"
            )

            if log_dir:
                log_llm_call(log_dir, call_info)

            raise e
