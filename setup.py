"""Install the git-theta package."""

import ast
import itertools
from pathlib import Path

from setuptools import find_packages, setup


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


# Packages to install for using different deep learning frameworks.
frameworks_require = {
    "pytorch": ["git_theta_checkpoints_pytorch"],
    "torch": ["git_theta_checkpoints_pytorch"],
    "tensorflow": ["git_theta_checkpoints_tensorflow"],
    "flax": ["git_theta_checkpoints_flax"],
    "safetensors": ["git_theta_checkpoints_safetensors"],
}


with open(Path(__file__).parent / "README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name="git_theta",
    version=get_version("git_theta/__init__.py"),
    description="Version control system for machine learning model checkpoints.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Colin Raffel",
    author_email="craffel@gmail.com",
    url="https://github.com/r-three/git-theta",
    packages=find_packages(),
    include_package_data=True,
    package_data={"git_theta": ["hooks/post-commit", "hooks/pre-push"]},
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Version Control",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="git vcs machine-learning",
    license="MIT",
    install_requires=[
        "GitPython",
        "gitdb",
        "tensorstore >= 0.1.14",
        "file-or-name",
        "six",
        "scipy",
        "numba",
        "msgpack",
        'importlib_resources; python_version < "3.9.0"',
        'importlib_metadata; python_version < "3.10.0"',
        'typing_extensions; python_version < "3.8.0"',
        "prompt_toolkit",
        "colorama",
    ],
    extras_require={
        **frameworks_require,
        # Install all framework deps with the all target.
        "test": ["pytest"],
        "all": list(set(itertools.chain(*frameworks_require.values()))),
        "docs": ["sphinx", "numpydoc"],
    },
    entry_points={
        "console_scripts": [
            "git-theta = git_theta.scripts.git_theta_cli:main",
            "git-theta-filter = git_theta.scripts.git_theta_filter:main",
            "git-theta-merge = git_theta.scripts.git_theta_merge:main",
            "git-theta-diff = git_theta.scripts.git_theta_diff:main",
        ],
        "git_theta.plugins.checkpoints": [],
        "git_theta.plugins.checkpoint.sniffers": [],
        "git_theta.plugins.updates": [
            "dense = git_theta.updates.dense:DenseUpdate",
            "sparse = git_theta.updates.sparse:SparseUpdate",
            "low-rank = git_theta.updates.low_rank:LowRankUpdate",
            "ia3 = git_theta.updates.ia3:IA3Update",
        ],
        "git_theta.plugins.merges": [
            "take_us = git_theta.merges.take:TakeUs",
            "take_them = git_theta.merges.take:TakeThem",
            "take_original = git_theta.merges.take:TakeOriginal",
            "average-ours-theirs = git_theta.merges.average:Average",
            "average-all = git_theta.merges.average:AverageAll",
            "average-ours-original = git_theta.merges.average:AverageOursOriginal",
            "average-theirs-original = git_theta.merges.average:AverageTheirsOriginal",
            "context = git_theta.merges.context:Context",
        ],
    },
)
