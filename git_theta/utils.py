"""Utilities for git theta."""

import dataclasses
import datetime
import functools
import inspect
import os
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from types import MethodType
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union


def _format(self, value, tag):
    """Wrap `value` in HTML like <tag>s."""
    return f"<{getattr(self, tag)}>{value}</{getattr(self, tag)}>"


# TODO: Make configurable
@dataclasses.dataclass
class TextStyle:
    param: str = "purple"
    model: str = "cyan"
    argument: str = "Salmon"
    who: str = "u"
    changed: str = "yellow"
    added: str = "green"
    deleted: str = "red"

    # TODO: Move this to classlevel code gen?
    def __post_init__(self):
        # Add format_field() methods to the object for each field.
        for field in dataclasses.fields(self):
            field = field.name
            method = MethodType(functools.partial(_format, tag=field), self)
            setattr(self, f"format_{field}", method)


TEXT_STYLE = TextStyle()


# TODO: Defer the creation of enum text so we don't need an instantiated
# TextStyle at import time.
class DiffState(Enum):
    EQUAL = "All parameter values are equal."
    CHANGED_A = f"{TEXT_STYLE.format_who('We')} <b>{TEXT_STYLE.format_changed('changed')}</b> this parameter."
    CHANGED_B = f"{TEXT_STYLE.format_who('They')} <b>{TEXT_STYLE.format_changed('changed')}</b> this parameter."
    CHANGED_BOTH = f"{TEXT_STYLE.format_who('Both')} them and us <b>{TEXT_STYLE.format_changed('changed')}</b> this parameter."
    DELETED_A = f"{TEXT_STYLE.format_who('We')} <b>{TEXT_STYLE.format_deleted('deleted')}</b> this parameter."
    DELETED_B = f"{TEXT_STYLE.format_who('They')} <b>{TEXT_STYLE.format_deleted('deleted')}</b> this parameter."
    DELETED_BOTH = f"{TEXT_STYLE.format_who('Both')} them and us <b>{TEXT_STYLE.format_deleted('deleted')}</b> this parameter."
    ADDED_A = f"{TEXT_STYLE.format_who('We')} <b>{TEXT_STYLE.format_added('added')}</b> this parameter."
    ADDED_B = f"{TEXT_STYLE.format_who('They')} <b>{TEXT_STYLE.format_added('added')}</b> this parameter."
    ADDED_BOTH = f"{TEXT_STYLE.format_who('Both')} them and us <b>{TEXT_STYLE.format_added('added')}</b> this parameter."


class NoResult:
    """A sentinel class used to mark no-action choices."""


@dataclass
class EnvVar:
    name: str
    default: Any

    def __get__(self, obj, objtype=None):
        value = os.environ.get(self.name)
        return type(self.default)(value) if value else self.default


class EnvVarConstants:
    CHECKPOINT_TYPE = EnvVar(name="GIT_THETA_CHECKPOINT_TYPE", default="sniff")
    UPDATE_TYPE = EnvVar(name="GIT_THETA_UPDATE_TYPE", default="dense")
    UPDATE_DATA_PATH = EnvVar(name="GIT_THETA_UPDATE_DATA_PATH", default="")
    PARAMETER_ATOL = EnvVar(name="GIT_THETA_PARAMETER_ATOL", default=1e-8)
    PARAMETER_RTOL = EnvVar(name="GIT_THETA_PARAMETER_RTOL", default=1e-5)
    LSH_SIGNATURE_SIZE = EnvVar(name="GIT_THETA_LSH_SIGNATURE_SIZE", default=16)
    LSH_THRESHOLD = EnvVar(name="GIT_THETA_LSH_THRESHOLD", default=1e-6)
    LSH_POOL_SIZE = EnvVar(name="GIT_THETA_LSH_POOL_SIZE", default=10_000)
    MAX_CONCURRENCY = EnvVar(name="GIT_THETA_MAX_CONCURRENCY", default=-1)
    MANUAL_MERGE = EnvVar(name="GIT_THETA_MANUAL_MERGE", default=False)
    LOG_LEVEL = EnvVar(name="GIT_THETA_LOG_LEVEL", default="DEBUG")
    LOW_MEMORY = EnvVar(name="GIT_THETA_LOW_MEMORY", default=False)


def flatten(
    d: Dict[str, Any],
    is_leaf: Callable[[Any], bool] = lambda v: not isinstance(v, dict),
) -> Dict[Tuple[str, ...], Any]:
    """Flatten a nested dictionary.

    Parameters
    ----------
    d:
        The nested dictionary to flatten.

    Returns
    -------
    Dict[Tuple[str, ...], Any]
        The flattened version of the dictionary where the key is now a tuple
        of keys representing the path of keys to reach the value in the nested
        dictionary.
    """

    def _flatten(d, prefix: Tuple[str] = ()):
        flat = type(d)({})
        for k, v in d.items():
            if not is_leaf(v):
                flat.update(_flatten(v, prefix=prefix + (k,)))
            else:
                flat[prefix + (k,)] = v
        return flat

    return _flatten(d)


def unflatten(d: Dict[Tuple[str, ...], Any]) -> Dict[str, Union[Dict[str, Any], Any]]:
    """Unflatten a dict into a nested one.

    Parameters
    ----------
    d:
        The dictionary to unflatten. Each key should be a tuple of keys the
        represent the nesting.

    Returns
    Dict
        The nested version of the dictionary.
    """
    nested = type(d)({})
    for ks, v in d.items():
        curr = nested
        for k in ks[:-1]:
            curr = curr.setdefault(k, {})
        curr[ks[-1]] = v
    return nested


def is_valid_oid(oid: str) -> bool:
    """Check if an LFS object-id is valid

    Parameters
    ----------
    oid:
        LFS object-id

    Returns
    bool
        Whether this object-id is valid
    """
    return re.match("^[0-9a-f]{64}$", oid) is not None


def is_valid_commit_hash(commit_hash: str) -> bool:
    """Check if a git commit hash is valid

    Parameters
    ----------
    commit_hash
        Git commit hash

    Returns
    bool
        Whether this commit hash is valid
    """
    return re.match("^[0-9a-f]{40}$", commit_hash) is not None


def remove_suffix(s: str, suffix: str) -> str:
    """Remove suffix matching copy semantics of methods in later pythons."""
    if suffix and s.endswith(suffix):
        return s[: -len(suffix)]
    return s[:]


class Trie:
    """Data structure for O(n) prefix existence checking."""

    def __init__(self, char: Optional[str] = None):
        self.char = char  # Really only for debugging.
        self.next: Dict[str, Trie] = {}
        self.is_word = False

    def insert(self, word: str):
        # If there are no more letters to add we must be a full word.
        if not word:
            self.is_word = True
            return
        # Get the node for the next character.
        first_char = word[0]
        # Explicit checks over something like ".get" to ensure we only make a
        # node if we have too. Also avoid defaultdict so we don't accidentally
        # create a node during search.
        if first_char not in self.next:
            node = self.__class__(first_char)
            self.next[first_char] = node
        else:
            node = self.next[first_char]
        suffix = word[1:]
        # Recurse
        node.insert(suffix)

    def _query(self, word: str) -> "Trie":
        """Find the Node associated with `word`."""
        # If there are no more characters we are the end of the line.
        if not word:
            return self
        # Get the next node, raise key error if prefix + char is not one we have
        # seen before.
        return self.next[word[0]]._query(word[1:])

    def prefix(self, word: str) -> bool:
        # Note, a full word needs to be part of a longer word to be a prefix.
        # i.e. It has characters that come after it.
        try:
            node = self._query(word)
            # Does the node have continuations?
            return bool(node.next)
        # We didn't make it until the end of the word.
        except KeyError:
            return False

    def __contains__(self, word: str) -> bool:
        # Note, this is O(n) so if there is a set of words it would be faster to
        # check that.
        try:
            node = self._query(word)
            # Does the node represent a word?
            return node.is_word
        # We didn't make it until the end of the word.
        except KeyError:
            return False

    @classmethod
    def from_iterable(cls, words: Iterable[str]) -> "Trie":
        """Build a trie for some collection of words."""
        root = cls()
        for word in words:
            root.insert(word)
        return root

    def __str__(self):
        return f"{self.__class__.__name__}(char={self.char}, is_word={self.is_word}, next={self.next.keys()})"


def abstract_classattributes(*attributes):
    """Force subclasses to define some class attributes."""

    # Note, we are modifying the class with the decorator, not wrapping it so we
    # don't need @functools.wraps to maintain information about our class.
    def inner(base_cls):
        # First set all attributes (on the base class) to NotImplemented
        # so the subclasses will inherit them and give us something to compare to.
        for attribute in attributes:
            setattr(base_cls, attribute, NotImplemented)

        # Save the original verson of the init, this lets us exectue it to make
        # sure we preserve any functionality defined by the base class.
        og_init_subclass = base_cls.__init_subclass__

        # A new version of __init_subclass__ that checks for our attributes.
        # TODO: How does this change things like __doc__ and __name__?
        def enforcing_init_subclass(cls, **kwargs):
            # Call the original one, if they didn't override the implementation it
            # doesn't take cls as an argument. Just try each version to see what works.
            try:
                og_init_subclass(cls, **kwargs)
            except TypeError:
                og_init_subclass(**kwargs)

            # Collect all attributes on the subclass now, that still have their
            # default (inherited) value.
            missing_attributes = []
            for attribute in attributes:
                if getattr(cls, attribute, NotImplemented) is NotImplemented:
                    missing_attributes.append(attribute)

            # Error out if they are a concrete class and missing things.
            if missing_attributes and not inspect.isabstract(cls):
                missing = [f'"{attr}"' for attr in missing_attributes]
                plural = ""
                # Make it look pretty and handle pluralization.
                if len(missing) > 1:
                    missing[-1] = f"and {missing[-1]}"
                    plural = "s"
                missing = ", ".join(missing)
                raise NotImplementedError(
                    f"Abstract Attribute{plural} {missing} missing "
                    f"on class {cls.__name__}"
                )

            return cls

        # Set this as the new __init_subclass__ and make sure it is a class method.
        base_cls.__init_subclass__ = classmethod(enforcing_init_subclass)
        return base_cls

    return inner


def touch(path: str):
    """Update the access and modify time of `path`."""
    dt_epoch = datetime.datetime.now().timestamp()
    os.utime(path, (dt_epoch, dt_epoch))
