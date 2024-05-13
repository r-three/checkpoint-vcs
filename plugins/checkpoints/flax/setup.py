"""Plugin to support the flax checkpoint format."""

import ast
import os

from setuptools import setup


def get_version(file_name: str, version_variable: str = "__version__") -> str:
    """Find the version by walking the AST to avoid duplication.

    Parameters
    ----------
    file_name : str
        The file we are parsing to get the version string from.
    version_variable : str
        The variable name that holds the version string.

    Raises
    ------
    ValueError
        If there was no assignment to version_variable in file_name.

    Returns
    -------
    version_string : str
        The version string parsed from file_name_name.
    """
    with open(file_name) as f:
        tree = ast.parse(f.read())
        # Look at all assignment nodes that happen in the ast. If the variable
        # name matches the given parameter, grab the value (which will be
        # the version string we are looking for).
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if node.targets[0].id == version_variable:
                    return node.value.s
    raise ValueError(
        f"Could not find an assignment to {version_variable} " f"within '{file_name}'"
    )


setup(
    name="git_theta_checkpoints_flax",
    description="Plugin to support the flax checkpoint format.",
    install_requires=[
        # "git_theta",
        "flax",
        "jax",
    ],
    version=get_version("git_theta_checkpoints_flax/__init__.py"),
    packages=[
        "git_theta_checkpoints_flax",
    ],
    author="Brian Lester",
    entry_points={
        "git_theta.plugins.checkpoints": [
            "flax = git_theta_checkpoints_flax.checkpoints:FlaxCheckpoint",
            "flax-checkpoint = git_theta_checkpoints_flax.checkpoints:FlaxCheckpoint",
        ],
        "git_theta.plugins.checkpoint.sniffers": [
            "flax = git_theta_checkpoints_flax.sniffer:flax_sniffer",
        ],
    },
)
