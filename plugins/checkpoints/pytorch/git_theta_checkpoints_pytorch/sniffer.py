"""Infer if a checkpoint is pytorch based.

We put this in a different file to avoid importing dl frameworks for file sniffing.
"""

import re


def pytorch_sniffer(checkpoint_path: str) -> bool:
    # Many checkpoints on HuggingFace Hub are named this.
    if checkpoint_path == "pytorch_model.bin":
        return True
    if re.search(r"\.py?t$", checkpoint_path):
        return True
    return False
