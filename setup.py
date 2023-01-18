"""Install the git-theta package."""

import ast
from setuptools import setup, find_packages


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
    name="git_theta",
    version=get_version("git_theta/__init__.py"),
    description="Version control system for model checkpoints.",
    author="Colin Raffel",
    author_email="craffel@gmail.com",
    url="https://github.com/r-three/checkpoint-vcs",
    packages=find_packages(),
    package_data={"git_theta": ["hooks/post-commit", "hooks/pre-push"]},
    scripts=["bin/git-theta", "bin/git-theta-filter"],
    long_description="Version control system for model checkpoints.",
    python_requires=">=3.7",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Version Control",
    ],
    keywords="git vcs machine-learning",
    license="MIT",
    install_requires=[
        "GitPython",
        "tensorstore >= 0.1.14",
        "file-or-name",
        "six",
        "scipy",
        'importlib_resources; python_version < "3.9.0"',
        'importlib_metadata; python_version < "3.10.0"',
        'typing_extensions; python_version < "3.8.0"',
    ],
    extras_require={
        "test": ["pytest"],
        "pytorch": ["torch"],
        "tensorflow": ["tensorflow"],
        # TODO: Is there a way for all to be the concat of the list of others
        # instead of making a new list? In case some framework needs multiple
        # libraries installed?
        "all": ["torch", "tensorflow"],
    },
    entry_points={
        "git_theta.plugins.checkpoints": [
            "pytorch = git_theta.checkpoints:PickledDictCheckpoint",
            "pickled-dict = git_theta.checkpoints:PickledDictCheckpoint",
            "tf = git_theta.checkpoints.tensorflow_checkpoint:TensorFlowCheckpoint",
            "tensorflow = git_theta.checkpoints.tensorflow_checkpoint:TensorFlowCheckpoint",
            "tensorflow-checkpoint = git_theta.checkpoints.tensorflow_checkpoint:TensorFlowCheckpoint",
            "tf-savedmodel = git_theta.checkpoints.tensorflow_checkpoint:TensorFlowSavedModel",
            "tensorflow-savedmodel = git_theta.checkpoints.tensorflow_checkpoint:TensorFlowSavedModel",
        ],
        "git_theta.plugins.updates": [
            "dense = git_theta.updates.dense:DenseUpdate",
            "sparse = git_theta.updates.sparse:SparseUpdate",
            "low-rank = git_theta.updates.low_rank:LowRankUpdate",
        ],
    },
)
