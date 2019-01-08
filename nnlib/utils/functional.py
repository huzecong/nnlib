from functools import reduce
from typing import Callable, Iterator, List, Sequence, TypeVar, overload, Iterable, Optional, Any

__all__ = ['scanl', 'scanr', 'is_none', 'not_none', 'filter_none', 'split_according_to', 'split_by']

A = TypeVar('A')
B = TypeVar('B')


# This is what happens when you don't have the Haskell `scanl`
@overload
def scanl(func: Callable[[A, A], A], lst: Sequence[A]) -> List[A]: ...


@overload
def scanl(func: Callable[[B, A], B], lst: Sequence[A], initial: B) -> List[B]: ...


def scanl(func, lst, initial=None):
    """
    Computes the intermediate results of :func:`reduce`. Equivalent to Haskell's ``scanl``. For example::

        >>> scanl(operator.add, [1, 2, 3, 4], 0)
        [0, 1, 3, 6, 10]

        >>> scanl(lambda s, x: x + s, ['a', 'b', 'c', 'd'])
        ['a', 'ba', 'cba', 'dcba']

    Learn more at `Learn You a Haskell: Higher Order Functions <http://learnyouahaskell.com/higher-order-functions>`_.

    :param func: The function to apply. This should be a binary function where the arguments are: the accumulator,
        and the current element.
    :param lst: The list of elements to iteratively apply the function to.
    :param initial: The initial value for the accumulator. If not supplied, the first element in the list is used.
    :return: The intermediate results at each step.
    """
    if initial is None:
        return reduce(lambda l, x: l.__iadd__([func(l[-1], x)]), lst[1:], [lst[0]])
    else:
        return reduce(lambda l, x: l.__iadd__([func(l[-1], x)]), lst, [initial])


@overload
def scanr(func: Callable[[A, A], A], lst: Sequence[A]) -> List[A]: ...


@overload
def scanr(func: Callable[[B, A], B], lst: Sequence[A], initial: B) -> List[B]: ...


def scanr(func, lst, initial=None):
    """
    Computes the intermediate results of :func:`reduce` applied in reverse. Equivalent to Haskell's ``scanr``.
    For example::

        >>> scanr(operator.add, [1, 2, 3, 4], 0)
        [10, 9, 7, 4, 0]

        >>> scanr(lambda s, x: x + s, ['a', 'b', 'c', 'd'])
        ['abcd', 'bcd', 'cd', 'd']

    Learn more at `Learn You a Haskell: Higher Order Functions <http://learnyouahaskell.com/higher-order-functions>`_.

    :param func: The function to apply. This should be a binary function where the arguments are: the accumulator,
        and the current element.
    :param lst: The list of elements to iteratively apply the function to.
    :param initial: The initial value for the accumulator. If not supplied, the first element in the list is used.
    :return: The intermediate results at each step, starting from the end.
    """
    if initial is None:
        return reduce(lambda l, x: l.__iadd__([func(l[-1], x)]), lst[-2::-1], [lst[-1]])[::-1]
    else:
        return reduce(lambda l, x: l.__iadd__([func(l[-1], x)]), lst[::-1], [initial])[::-1]


def is_none(x: Optional[Any]) -> bool:
    """
    Returns whether the argument is ``None``.
    """
    return x is None


def not_none(x: Optional[Any]) -> bool:
    """
    Returns whether the argument is not ``None``.
    """
    return x is not None


def filter_none(x: Iterable[Optional[A]]) -> Iterable[A]:
    """
    Filters not-\ ``None`` elements in list. Returns a generator.
    """
    return filter(not_none, x)


def split_according_to(criterion: Callable[[A], bool], _list: Sequence[A], empty_segments=False) \
        -> Iterator[Sequence[A]]:
    """
    Find elements in list where ``criterion`` is satisfied, and split list into sub-lists by dropping those elements.

    :type criterion: (Any) -> bool
    :param criterion: The criterion to satisfy.
    :param _list: The list to split.
    :param empty_segments: If ``True``, include an empty list in cases where two adjacent elements satisfy
        the criterion.
    :return: List of sub-lists.
    """
    last = 0
    for i in range(len(_list)):
        if criterion(_list[i]):
            if last < i or empty_segments:
                yield _list[last:i]
            last = i + 1
    if last < len(_list) or empty_segments:
        yield _list[last:]


def split_by(sep: A, _list: Sequence[A], empty_segments=False) -> Iterator[Sequence[A]]:
    """
    Split list into sub-lists by dropping elements that matches ``sep``.

    :param sep: Separator token.
    :param _list: The list to split.
    :param empty_segments: If ``True``, include an empty list in cases where two adjacent elements satisfy
        the criterion.
    :return: List of sub-lists.
    """
    yield from split_according_to(lambda x: x == sep, _list, empty_segments)
