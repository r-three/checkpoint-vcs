"""Infer if a checkpoint is flax based.

We put this in a different file to avoid importing dl frameworks for file sniffing.
"""


def flax_sniffer(checkpoint_path: str) -> bool:
    # TODO: Check if the actual value is msgpack based on magic numbers?
    return checkpoint_path.endswith(".flax")
