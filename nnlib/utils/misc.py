"""
Miscellaneous Functions
"""
import functools
import io
from typing import Dict, Iterable, List, TypeVar, Union, overload

import numpy as np

from nnlib.utils.logging import Logging

__all__ = ['progress', 'deprecated', 'map_to_list', 'memoization', 'reverse_map']

T = TypeVar('T')


@overload  # type: ignore
def progress(iterable: Iterable[T], verbose=True, **kwargs) -> Iterable[T]:
    ...


@overload
def progress(iterable: int, verbose=True, **kwargs) -> Iterable[int]:
    ...


class _DummyTqdm:
    r""" A sequence wrapper that ignores everything """

    def __init__(self, iterable):
        self.iterable = iterable  # could be None

    def __next__(self):
        return next(self.iterable)

    def __iter__(self):
        return _DummyTqdm(iter(self.iterable))

    @staticmethod
    def nop(*args, **kwargs):
        pass

    def __getattr__(self, item):
        return _DummyTqdm.nop


try:
    import os
    import typing

    # noinspection PyUnresolvedReferences
    import tqdm


    def progress(iterable=None, verbose=True, **kwargs):  # type: ignore
        if not verbose:
            return _DummyTqdm(iterable)  # could be none
        # bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} ~{remaining}]'
        if isinstance(iterable, int):
            return tqdm.trange(iterable, **kwargs)
        if iterable is None:
            return tqdm.tqdm(**kwargs)
        if isinstance(iterable, io.TextIOBase):
            # noinspection PyUnresolvedReferences
            from .io import FileProgressWrapper  # avoid circular import
            return FileProgressWrapper(iterable, verbose=verbose, **kwargs)
        return tqdm.tqdm(iterable, **kwargs)

except ImportError:
    # noinspection PyUnusedLocal
    def progress(iterable=None, verbose=True, **kwargs):  # type: ignore
        if not verbose:
            return _DummyTqdm(iterable)
        Logging.warn("`tqdm` package is not installed, no progress bar is shown.", category=ImportWarning)
        if type(iterable) is int:
            return range(iterable)
        return iterable


def deprecated(new_func=None):
    def decorator(func):
        warn_msg = f"{func.__name__} is deprecated."
        if new_func is not None:
            warn_msg += f" Use {new_func.__name__} instead."

        def wrapped(*args, **kwargs):
            Logging.warn(warn_msg, category=DeprecationWarning)
            return func(*args, **kwargs)

        return wrapped

    return decorator


def map_to_list(d: Dict[int, T]) -> List[T]:
    r"""
    Given a dict mapping indices (continuous indices starting from 0) to values, convert it into a list.

    :type d: dict
    """
    return [d[idx] for idx in range(len(d))]


def memoization(f):
    @functools.wraps(f)
    def wrapped(*args):
        key = tuple(args)
        if key in wrapped._states:
            return wrapped._states[key]
        ret = f(*args)
        wrapped._states[key] = ret
        return ret

    wrapped._states = {}
    return wrapped


@overload
def reverse_map(d: Dict[T, int]) -> List[T]: ...


@overload
def reverse_map(d: Union[np.ndarray, List[int]]) -> List[int]: ...


def reverse_map(d):
    r"""
    Given a dict containing pairs of (`item`, `id`), return a list where the `id`-th element is `item`.

    Or, given a list containing a permutation, return its reverse.

    :type d: dict | list | np.ndarray
    """
    if isinstance(d, dict):
        return [k for k, _ in sorted(d.items(), key=lambda xs: xs[1])]

    rev = [0] * len(d)
    for idx, x in enumerate(d):
        rev[x] = idx
    return rev
