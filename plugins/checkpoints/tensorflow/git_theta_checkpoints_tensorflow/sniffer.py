"""Infer if a checkpoint is tensorflow based.

We put this in a different file to avoid importing dl frameworks for file sniffing.
"""


def tensorflow_sniffer(checkpoint_path: str) -> bool:
    return checkpoint_path.endswith(".tf")


# TODO: Add support for detecting saved models.
def saved_model_sniffer(checkpoint_path: str) -> bool:
    # We don't support saved models yet.
    return False
