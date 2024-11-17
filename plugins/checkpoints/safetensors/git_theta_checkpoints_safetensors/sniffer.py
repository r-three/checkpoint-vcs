"""Infer if a checkpoint is safetensors based.

We put this in a different file to avoid importing dl frameworks for file sniffing.
"""


def safetensors_sniffer(checkpoint_path: str) -> bool:
    return checkpoint_path.endswith(".safetensors")
