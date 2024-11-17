"""Base class and utilities for different checkpoint format backends."""

import os
import sys
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from git_theta import utils


@utils.abstract_classattributes("name")
class Checkpoint(dict, metaclass=ABCMeta):
    """Abstract base class for wrapping checkpoint formats."""

    name: str = NotImplemented  # The name of this checkpoint handler, can be used to lookup the plugin.

    @classmethod
    def from_file(cls, checkpoint_path):
        """Create a new Checkpoint object.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to a checkpoint file
        """
        return cls.from_framework(cls.load(checkpoint_path))

    @classmethod
    @abstractmethod
    def load(cls, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to a checkpoint file

        Returns
        -------
        model_dict : dict
            Dictionary mapping parameter names to parameter values.  Parameters
            should be numpy arrays.
        """

    @classmethod
    @abstractmethod
    def from_framework(cls, model_dict):
        """Convert a checkpoint from the native framework format to git-thetas."""

    @abstractmethod
    def to_framework(self):
        """Convert out checkpoint into the native framework state dict."""

    @abstractmethod
    def save(self, checkpoint_path):
        """Load a checkpoint into a dict format.

        Parameters
        ----------
        checkpoint_path : str or file-like object
            Path to write out the checkpoint file to
        """

    def flatten(self):
        return utils.flatten(self, is_leaf=lambda v: isinstance(v, np.ndarray))

    def unflatten(self):
        return utils.unflatten(self)

    @classmethod
    def diff(cls, m1: "Checkpoint", m2: "Checkpoint") -> "Checkpoint":
        """Compute the diff between two checkpoints.

        Parameters
        ----------
        m1 : Checkpoint
            The new checkpoint
        m2 : Checkpoint
            The old checkpoint

        Returns
        -------
        added : Checkpoint
            Checkpoint containing the parameter groups added to m1
        removed : Checkpoint
            Checkpoint containing the parameter groups removed from m2
        modified : Checkpoint
            Checkpoint containing the parameter groups modified between m1 and m2
        """
        m1_flat = m1.flatten()
        m2_flat = m2.flatten()
        # N.b.: This is actually faster than set operations on m1 and m2's keys
        added = cls({k: v for k, v in m1_flat.items() if k not in m2_flat}).unflatten()
        removed = cls(
            {k: v for k, v in m2_flat.items() if k not in m1_flat}
        ).unflatten()
        modified = cls(
            {
                k: v
                for k, v in m1_flat.items()
                if k in m2_flat and not np.allclose(v, m2_flat[k])
            }
        ).unflatten()
        return added, removed, modified


def get_checkpoint_handler_name(checkpoint_type: Optional[str] = None) -> str:
    """Get the name of the checkpoint handler to use.

    Order of precedence is
    1. `checkpoint_type` argument
    2. `$GIT_THETA_CHECKPOINT_TYPE` environment variable
    3. default value (currently pytorch)

    Parameters
    ----------
    checkpoint_type
        Name of the checkpoint handler

    Returns
    -------
    str
        Name of the checkpoint handler
    """
    # TODO(bdlester): Find a better way to include checkpoint type information
    # in git clean filters that are run without `git theta add`.
    # TODO: Don't default to pytorch once other checkpoint formats are supported.
    return checkpoint_type or utils.EnvVarConstants.CHECKPOINT_TYPE


def get_checkpoint_handler(checkpoint_type: Optional[str] = None) -> Checkpoint:
    """Get the checkpoint handler either by name or from an environment variable.

    Gets the checkpoint handler either for the `checkpoint_type` argument or
    `$GIT_THETA_CHECKPOINT_TYPE` environment variable.

    Defaults to pytorch when neither are defined.

    Parameters
    ----------
    checkpoint_type
        Name of the checkpoint handler

    Returns
    -------
    Checkpoint
        The checkpoint handler (usually an instance of `git_theta.checkpoints.Checkpoint`).
        Returned handler may be defined in a user installed plugin.
    """
    checkpoint_type = get_checkpoint_handler_name(checkpoint_type)
    discovered_plugins = entry_points(group="git_theta.plugins.checkpoints")
    return discovered_plugins[checkpoint_type].load()


def sniff_checkpoint(checkpoint_path) -> str:
    """En"""
    discovered_plugins = entry_points(group="git_theta.plugins.checkpoint.sniffers")
    loaded_plugins = {ep.name: ep.load() for ep in discovered_plugins}
    logger = logging.getLogger("git_theta")
    logger.debug(
        f"Sniffing {checkpoint_path} to infer which deep learning framework it is."
    )
    for ckpt_type, ckpt_sniffer in loaded_plugins.items():
        logger.debug(f"Checking if {checkpoint_path} is a {ckpt_type} checkpoint.")
        if ckpt_sniffer(checkpoint_path):
            logger.debug(
                f"Determined that {checkpoint_path} is a {ckpt_type} checkpoint."
            )
            return ckpt_type
    raise ValueError(f"Couldn't determine checkpoint type for {checkpoint_path}")
