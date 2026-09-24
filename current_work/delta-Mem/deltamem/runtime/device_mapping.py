"""Explicit, GPU-only layer placement for supported Delta-Mem backbones."""


def layerwise_device_map(num_layers: int, num_devices: int) -> dict[str, int]:
    """Keep decoder blocks intact and use every visible CUDA device.

    Embeddings and the output head share GPU 0, including for tied weights.
    This partitions by layer count, not by bytes or free GPU memory.
    """
    if num_devices < 2:
        raise ValueError("layerwise inference requires at least two visible CUDA GPUs")
    if num_layers < num_devices:
        raise ValueError("layerwise inference requires at least one layer per GPU")
    return {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": num_devices - 1,
        "lm_head": 0,
        **{
            f"model.layers.{index}": index * num_devices // num_layers
            for index in range(num_layers)
        },
    }
