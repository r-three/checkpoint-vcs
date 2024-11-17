"""Checkpoint using the HF safetensors format.

safetensors has the ability to write model checkpoint from "dl-native" -> "safetensors"
and read "safetensors" -> any "dl-native" framework, not just the one that was
used to write it. Therefore, we read/write with their numpy API.
"""

import safetensors.numpy
from file_or_name import file_or_name

from git_theta.checkpoints import Checkpoint


# TODO(bdlester): Can we leverage the lazying loading ability to make things faster?
class SafeTensorsCheckpoint(Checkpoint):
    """Class for r/w of the safetensors format. https://github.com/huggingface/safetensors"""

    name: str = "safetensors"

    @classmethod
    @file_or_name(checkpoint_path="rb")
    def load(cls, checkpoint_path: str):
        # Note that we use the numpy as the framework because we don't care what
        # their downstream dl framework is, we only want the results back as
        # numpy arrays.
        return safetensors.numpy.load(checkpoint_path.read())

    @file_or_name(checkpoint_path="wb")
    def save(self, checkpoint_path: str):
        # Note, git theta uses numpy internally, so we save using the numpy api,
        # regardless of the original framework they used to write the checkpoint.
        checkpoint_dict = self.to_framework()
        checkpoint_path.write(safetensors.numpy.save(checkpoint_dict))

    def to_framework(self):
        return self

    @classmethod
    def from_framework(cls, model_dict):
        return cls(model_dict)
