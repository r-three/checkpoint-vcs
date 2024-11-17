"""A Checkpoint backend for tensorflow models."""


import numpy as np
import tensorflow as tf

from git_theta import utils
from git_theta.checkpoints import Checkpoint


class DynamicNetwork(tf.keras.Model):
    """A keras model that can dynamically build itself from a map of params for tf saving."""

    def __init__(self, params):
        super().__init__()
        for name, param in params.items():
            # Convert numpy to tf.Variable so it will be saved.
            if isinstance(param, np.ndarray):
                param = tf.Variable(param, name=name)
            # Converted nested models into nested networks.
            elif isinstance(param, dict):
                param = DynamicNetwork(param)
            # Save the variable (or sub-model) to an attribute so it will get tracked.
            self.__setattr__(name, param)


class TensorFlowCheckpoint(Checkpoint):
    """Process a TensorFlow checkpoint via `tf.keras.Model.save_weights`. (no computation graph included)."""

    name: str = "tensorflow-checkpoint"
    VALUE_STRING = ".ATTRIBUTES/VARIABLE_VALUE"

    @staticmethod
    def is_parameter(param_name: str) -> bool:
        return param_name.endswith(TensorFlowCheckpoint.VALUE_STRING)

    @staticmethod
    def normalize_name(param_name: str) -> str:
        param_name = utils.remove_suffix(param_name, TensorFlowCheckpoint.VALUE_STRING)
        param_name = utils.remove_suffix(param_name, "/")
        return param_name

    @classmethod
    def load(cls, checkpoint_path: str):
        ckpt_read = tf.train.load_checkpoint(checkpoint_path)
        params = {}
        for param_name in ckpt_read.get_variable_to_shape_map():
            if not TensorFlowCheckpoint.is_parameter(param_name):
                continue
            simple_name = TensorFlowCheckpoint.normalize_name(param_name)
            params[tuple(simple_name.split("/"))] = ckpt_read.get_tensor(param_name)
        return utils.unflatten(params)

    @classmethod
    def from_framework(cls, model_dict):
        return cls(model_dict)

    def to_framework(self):
        return self

    def save(self, checkpoint_path: str):
        model = DynamicNetwork(self)
        model.save_weights(checkpoint_path)


# TODO
class TensorFlowSavedModel(Checkpoint):
    """Process a TensorFlow SavedModel (computation graph included)."""

    name: str = "tensorflow-savedmodel"

    @classmethod
    def load(cls, checkpoint_path: str):
        raise ValueError("Sorry, SavedModel support is a work in progress.")

    def save(self, checkpoint_path: str):
        raise ValueError("Sorry, SavedModel support is a work in progress.")
