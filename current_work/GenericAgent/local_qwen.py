"""Local Transformers backend for GenericAgent's existing ToolClient."""
import threading

from llmcore import BaseSession, _fix_messages, _msgs_claude2oai


def text_messages(messages):
    result = []
    for message in messages:
        message = dict(message)
        content = message.get("content", "")
        if isinstance(content, list):
            if any(block.get("type") != "text" for block in content):
                raise ValueError("This local Qwen3 checkpoint supports text only")
            message["content"] = "\n".join(block["text"] for block in content)
        result.append(message)
    return result


class LocalQwenSession(BaseSession):
    def __init__(self, cfg):
        super().__init__({"apikey": "", "apibase": "", **cfg})
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(cfg.get("device", "cuda:0"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True)
        self.network = AutoModelForCausalLM.from_pretrained(
            self.model, local_files_only=True,
            torch_dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
            attn_implementation="sdpa",
        ).to(self.device).eval()
        self.max_input_tokens = cfg.get("max_input_tokens", 16384)
        self.max_tokens = cfg.get("max_tokens", 1536)
        self.do_sample = cfg.get("do_sample", True)
        self.top_p, self.top_k = cfg.get("top_p", 0.9), cfg.get("top_k", 0)
        self.seed = cfg.get("seed", 42)
        self.request_seed = self.seed
        self.request_calls = 0
        self.usage = []
        self.generate_lock = threading.Lock()

    def make_messages(self, raw_list):
        return _msgs_claude2oai(_fix_messages(raw_list))

    def raw_ask(self, messages):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        session = self

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return bool(getattr(session, "should_stop", lambda: False)())

        # Trim only complete old turns, never the current request.
        messages = text_messages(messages)
        while True:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            tokens = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
            if tokens.input_ids.shape[1] <= self.max_input_tokens:
                break
            first = 1 if messages[0]["role"] == "system" else 0
            if len(messages) - first <= 2:
                raise ValueError("Current request exceeds local max_input_tokens")
            del messages[first:first + 2]
        options = dict(max_new_tokens=self.max_tokens, do_sample=self.do_sample,
                       pad_token_id=self.tokenizer.eos_token_id, use_cache=True,
                       stopping_criteria=StoppingCriteriaList([Cancelled()]))
        if self.do_sample:
            options.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        seed = (self.request_seed + self.request_calls) % (2**63)
        with self.generate_lock, torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            output = self.network.generate(**tokens.to(self.device), **options)
        ids = output[0, tokens.input_ids.shape[1]:].tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        self.usage.append({"input_tokens": tokens.input_ids.shape[1],
                           "output_tokens": len(ids), "seed": seed,
                           "trimmed_messages": len(messages)})
        self.request_calls += 1
        yield text
        return [{"type": "text", "text": text}]
