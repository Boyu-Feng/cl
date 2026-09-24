from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .core import digest, save


def fingerprint(model, adapter=False):
    import hashlib
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if ("lora_" in name) == adapter:
            h.update(name.encode())
            # Hash all bytes, including frozen weights; do not rely on a sparse sample.
            values = p.detach().contiguous().view(torch.uint8).cpu()
            h.update(values.numpy().tobytes())
    return h.hexdigest()


def clipped_loss(selected, old, advantage):
    ratio = (selected - old).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()


class Writer:
    def __init__(self, plan, adapter=None, train=False):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.plan, self.train_enabled = plan, train
        torch.manual_seed(plan["train_seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(plan["model"], local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(plan["model"], torch_dtype=torch.bfloat16,
                attn_implementation="sdpa", local_files_only=True).to("cuda:0")
        if adapter:
            self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=train)
        elif train:
            self.model = get_peft_model(base, LoraConfig(r=plan["rank"], lora_alpha=plan["rank"] * 2,
                lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM", bias="none"))
        else:
            self.model = base.requires_grad_(False)
        self.model.eval()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if train:
            assert self.parameters and all("lora_" in n for n,p in self.model.named_parameters() if p.requires_grad)
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
            self.optimizer = torch.optim.AdamW(self.parameters, lr=plan["learning_rate"], weight_decay=0.0)
        self.base_before = fingerprint(self.model)
        self.adapter_before = fingerprint(self.model, True)

    def generate(self, messages_batch, random_seed, output_dirs):
        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in messages_batch]
        batch = self.tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
        width = batch.input_ids.shape[1]
        if width + self.plan["writer_max_tokens"] > self.plan["writer_context_limit"]:
            raise ValueError("Writer input exceeds context budget; no silent truncation")
        self.model.eval()
        started = time.monotonic()
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            output = self.model.generate(**batch.to("cuda:0"), do_sample=True,
                temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                max_new_tokens=self.plan["writer_max_tokens"], use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id, return_dict_in_generate=True,
                output_scores=True, use_model_defaults=False)
            tokens = output.sequences[:, width:]
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            selected = torch.stack([F.log_softmax(v.float(), dim=-1).gather(1, tokens[:,k:k+1]).squeeze(1)
                                    for k,v in enumerate(output.scores)], dim=1)
            samples = []
            for i, (prompt, messages, directory) in enumerate(zip(prompts, messages_batch, output_dirs)):
                response = tokens[i].tolist()
                stop = next((j + 1 for j,x in enumerate(response) if x in eos), len(response))
                response = response[:stop]
                prefix = batch.input_ids[i][batch.attention_mask[i].bool()].tolist()
                sample = {"input_ids": prefix + response, "prompt_length": len(prefix),
                          "old_logp": selected[i,:stop].cpu().tolist(),
                          "text": self.tokenizer.decode(response, skip_special_tokens=True).strip(),
                          "finish_reason": "stop" if response[-1] in eos else "length",
                          "response_tokens": stop, "messages": messages, "seed": random_seed,
                          "batch_index": i, "prompt_sha256": digest(prompt),
                          "batch_seconds": time.monotonic() - started}
                save(Path(directory) / "writer.json", sample)
                samples.append(sample)
        return samples

    def distribution(self, sample):
        ids = torch.tensor([sample["input_ids"]], device="cuda:0")
        length = ids.shape[1] - sample["prompt_length"]
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                            logits_to_keep=length + 1).logits[0,:-1].float()
        logp = F.log_softmax(logits, dim=-1)
        selected = logp.gather(1, ids[0,sample["prompt_length"]:,None]).squeeze(1)
        return logp, selected

    def update(self, samples):
        assert self.train_enabled
        started = time.monotonic()
        metrics = []
        # Two PPO epochs; advantages are EXACT supplied rewards, no group centering
        # or normalization. The delta baseline therefore remains in the objective.
        for epoch in range(2):
            self.optimizer.zero_grad(set_to_none=True)
            policy_values, kl_values, mismatches = [], [], []
            for sample in samples:
                self.model.eval()
                with torch.no_grad(), self.model.disable_adapter():
                    reference, _ = self.distribution(sample)
                self.model.train()
                logp, selected = self.distribution(sample)
                old = torch.tensor(sample["old_logp"], device="cuda:0")
                mismatch = float((selected.detach() - old).abs().mean())
                mismatches.append(mismatch)
                if epoch == 0 and mismatch > 0.15:
                    raise ValueError(f"Rollout/training logprob mismatch: {mismatch}")
                policy = clipped_loss(selected, old, sample["advantage"])
                kl = (logp.exp() * (logp - reference)).sum(-1).mean()
                loss = (policy + self.plan["kl_beta"] * kl) / len(samples)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                policy_values.append(float(policy.detach()))
                kl_values.append(float(kl.detach()))
                loss.backward()
                del reference, logp, selected, policy, kl, loss
            norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True))
            if not math.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient norm")
            self.optimizer.step()
            metrics.append({"epoch": epoch, "policy_loss": sum(policy_values)/len(samples),
                            "reference_kl": sum(kl_values)/len(samples), "gradient_norm": norm,
                            "max_rollout_logprob_mae": max(mismatches)})
        self.model.eval()
        return {"epochs": metrics, "seconds": time.monotonic()-started,
                "positive": sum(s["advantage"] > 0 for s in samples),
                "negative": sum(s["advantage"] < 0 for s in samples),
                "zero": sum(s["advantage"] == 0 for s in samples),
                "advantage_mean": sum(s["advantage"] for s in samples)/len(samples),
                "delta_mean": sum(s["delta"] for s in samples)/len(samples)}

    def save(self, out):
        out = Path(out)
        self.model.save_pretrained(out / "adapter")
        self.tokenizer.save_pretrained(out / "adapter")
        after = fingerprint(self.model)
        assert after == self.base_before, "Frozen base parameters changed"
        result = {"base_fingerprint_before": self.base_before, "base_fingerprint_after": after,
                  "base_unchanged": True, "adapter_before": self.adapter_before,
                  "adapter_after": fingerprint(self.model, True),
                  "trainable_parameters": sum(p.numel() for p in self.parameters)}
        save(out / "model_audit.json", result)
        return result
