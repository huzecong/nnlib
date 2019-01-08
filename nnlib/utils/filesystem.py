import os
from pathlib import Path
from typing import Union

__all__ = ['PathType', 'path_lca', 'path_add_suffix']

PathType = Union[str, Path]  # a union type for all possible paths


def path_lca(this: Path, other: PathType) -> Path:
    """
    Return the `lowest common ancestor <https://en.wikipedia.org/wiki/Lowest_common_ancestor>`_ of two paths.
    For example::

        >>> path_lca(Path("/path/to/file/in/here"), "/path/to/another/file")
        Path("/path/to/")

    **Caveat:** This implementation simply takes the longest common prefix of two expanded paths using
    :func:`os.path.commonprefix`, and may not be robust enough for complex use cases.

    :param this: The first path. Has to be of type :class:`pathlib.Path`.
    :param other: The second path. Can be either a :class:`pathlib.Path` or a :class:`str`.
    :return: The path to the LCA of two paths.
    """
    return this.relative_to(os.path.commonprefix([this, other]))


def path_add_suffix(this: Path, suffix: str) -> Path:
    """
    Append a suffix to the given path. For example::

        >>> path_add_suffix(Path("/path/to/file.txt", "bak"))
        Path("/path/to/file.txt.bak")

    :param this: The path to modify.
    :param suffix: The suffix to append.
    :return: The modified path.
    """
    suffix = suffix.strip()
    if suffix == '':
        return this.with_suffix(this.suffix)  # return a copy
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    return this.with_suffix(this.suffix + suffix)
