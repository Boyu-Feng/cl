"""Frozen local model backend shared by text-memory methods."""
import hashlib


class ContextLimitError(ValueError):
    pass


def check_context(input_tokens, output_tokens, limit):
    if input_tokens + output_tokens > limit:
        raise ContextLimitError(
            f"Full history needs {input_tokens} input + {output_tokens} reserved output tokens; "
            f"context limit is {limit}. No history was truncated.")


class LocalQwen:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.device = torch.device(args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, torch_dtype=getattr(torch, args.dtype),
            attn_implementation="sdpa").to(self.device).eval()
        self.model.requires_grad_(False)
        native_limit = self.model.config.max_position_embeddings
        self.context_limit = min(args.context_limit or native_limit, native_limit)

    def generate(self, messages, seed, *, max_new_tokens=None, temperature=None):
        import torch

        limit = self.args.max_new_tokens if max_new_tokens is None else max_new_tokens
        temperature = self.args.temperature if temperature is None else temperature
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        batch = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt", truncation=False)
        count = batch.input_ids.shape[1]
        check_context(count, limit, self.context_limit)
        options = dict(do_sample=temperature > 0, max_new_tokens=limit,
                       pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
        if temperature > 0:
            options.update(temperature=temperature, top_p=self.args.top_p, top_k=self.args.top_k)
        devices = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                   if self.device.type == "cuda" else [])
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            output = self.model.generate(**batch.to(self.device), **options)
        response_ids = output[0, count:].tolist()
        eos_ids = self.model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        ended = bool(response_ids) and response_ids[-1] in eos_ids
        return {"raw_response": self.tokenizer.decode(response_ids, skip_special_tokens=True).strip(),
                "input_tokens": count, "output_tokens": len(response_ids),
                "context_limit": self.context_limit,
                "finish_reason": "stop" if ended else "length",
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest()}
