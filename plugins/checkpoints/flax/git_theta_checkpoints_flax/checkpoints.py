"""Checkpoint plugin for Flax's msgpack format."""

from file_or_name import file_or_name
from flax import serialization

from git_theta.checkpoints import Checkpoint


class FlaxCheckpoint(Checkpoint):
    """Load a msgpack based Flax Checkpoint."""

    name: str = "flax"

    @classmethod
    @file_or_name(checkpoint_path="rb")
    def load(cls, checkpoint_path, **kwargs):
        return serialization.msgpack_restore(checkpoint_path.read())

    @classmethod
    def from_framework(cls, model_dict):
        return cls(model_dict)

    def to_framework(self):
        return self

    @file_or_name(checkpoint_path="wb")
    def save(self, checkpoint_path, **kwargs):
        checkpoint_path.write(serialization.msgpack_serialize(dict(self)))
