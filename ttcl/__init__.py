"""Isolated continual-learning methods and shared benchmark utilities."""

__all__ = ["OnlineLoRAMemory", "load_jsonl"]

def __getattr__(name):
    if name in ("OnlineLoRAMemory", "load_jsonl"):
        from . import online_lora
        return getattr(online_lora, name)
    raise AttributeError(name)
