"""
Math Utilities
"""
import collections

import numpy as np

from .functional import not_none

__all__ = ['FNVHash', 'ceil_div', 'normalize', 'pow', 'prod', 'sum', 'random_subset', 'softmax']


class FNVHash:
    hval = 0x811c9dc5
    fnv_32_prime = 0x01000193
    uint32_max = 2 ** 32

    @staticmethod
    def hash(s):
        h = FNVHash.hval
        for ch in s:
            h = ((h ^ ord(ch)) * FNVHash.fnv_32_prime) % FNVHash.uint32_max
        return h


def ceil_div(a, b):
    r"""
    Integer division that rounds up.
    """
    return (a + b - 1) // b


def normalize(xs):
    r"""
    NumPy-based normalization.
    """
    arr = np.asarray(xs, dtype=np.float)
    return arr / np.sum(arr)


# noinspection PyShadowingBuiltins
def pow(a, b, fast=True):
    r"""
    Compute ``a ** b`` (``a`` raised to ``b``-th power). ``b`` has to be a positive integer.

    **Note:** It is not required for ``type(a)`` to have an identity element.

    :param a: The base. Can be any variable that supports multiplication ``*``.
    :type b: int
    :param b: The exponent.
    :param fast: Whether to use fast exponent algorithm (that runs in logarithmic time).
    """
    if type(b) is not int:
        raise TypeError("Exponent should be a positive integer.")
    if b < 1:
        raise ValueError("Exponent should be a positive integer.")
    result = a
    b -= 1
    if fast:  # O(log b)
        while b > 0:
            if b % 2 == 1:
                result *= a
                b //= 2
            a *= a
    else:  # O(b)
        while b > 0:
            result *= a
            b -= 1
    return result


def _reduce(fn, *args):
    r"""
    Recursively reduce over sequence of values, where values can be sequences. None values are ignored.

    :type fn: (Any, Any) -> Any
    :param fn: Function taking (accumulator, element) and returning new accumulator.
    """
    result = None
    for x in filter(not_none, args):
        val = _reduce(fn, *x) if isinstance(x, collections.Iterable) else x
        if result is None:
            result = val
        else:
            result = fn(result, val)
    return result


def _ireduce(fn, *args):
    r"""
    In-place version of ``_reduce``.

    :type fn: (Any, Any) -> None
    :param fn: Function taking (accumulator, element) and performing in-place operation on accumulator.
    """
    return _reduce(lambda x, y: [fn(x, y), x][-1], *args)


def prod(*args):
    r"""
    Compute product of arguments, ignoring ``None`` values. Arguments could contain lists, or list of lists, etc.

    **Note:** It is not required for list elements to have an identity element.
    """
    return _reduce(lambda x, y: x.__rmul__(y), *args)


# noinspection PyShadowingBuiltins
def sum(*args):
    r"""
    Compute sum of arguments, ignoring ``None`` values. Arguments could contain lists, or list of lists, etc.

    **Note:** It is not required for list elements to have an identity element.
    """
    return _reduce(lambda x, y: x.__add__(y), *args)


def random_subset(total, size):
    r"""
    Select a random subset of size ``size`` from the larger set of size ``total``.

    This method is implemented to replace :func:`numpy.random.choice` with :attr:`replacement=False`.

    :param total: Size of the original set.
    :param size: Size of the randomly selected subset.
    :return: The 0-based indices of the subset elements.
    """
    if type(size) is float:
        size = int(total * size)
    # Don't trust `np.random.choice` without replacement! It's using brute force!
    if size * np.log(size) > total:
        return np.random.permutation(np.arange(total))[:size]
    else:
        choices = set()
        while len(choices) < size:
            choices.add(np.random.choice(total, size - len(choices)).tolist())
        return choices


def softmax(xs, t=1):
    r"""
    NumPy-based softmax with temperature. Returns a sequence with each element calculated as:

    .. math::
        s_i = \frac{ \exp(x_i / t) }{ \sum_x \exp(x / t) }

    :param xs: The sequence of weights.
    :param t: Temperature. Higher temperatures give a more uniform distribution, while lower temperatures give a more
        peaked distribution.
    """
    arr = np.exp(np.asarray(xs) / t)
    return arr / np.sum(arr)
