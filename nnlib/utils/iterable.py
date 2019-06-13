"""
List, Iterator, and Lazy Evaluation Utilities
"""
from typing import Any, Callable, Generic, Iterable, Iterator, List, Optional, Sequence, TypeVar

import numpy as np

__all__ = ['flat_iter', 'LazyList', 'ListWrapper', 'MeasurableGenerator', 'Range']

T = TypeVar('T')


def flat_iter(lst: Sequence[Any], expand_str: bool = False) -> Iterator[Any]:
    r"""
    Recursively flatten list elements. For example::

        >>> list(flat_iter([[1, 2, 3], [[[4]]], 5, 6, ['abcd']], expand_str=True))
        [1, 2, 3, 4, 5, 6, 'a', 'b', 'c', 'd']

        >>> list(flat_iter([[1, 2, 3], [[[4]]], 5, 6, ['abcd']], expand_str=False))
        [1, 2, 3, 4, 5, 6, 'abcd']

    :param lst: The list to flatten.
    :param expand_str: If ``True``, list elements of type :class:`str` are further expanded to
        singleton :class:`str`\ s.
    :return: A generator yielding flattened list elements.
    """
    if type(lst) is str:
        if expand_str:
            yield from lst
        else:
            yield lst
    else:
        for x in lst:
            if hasattr(x, '__getitem__') or hasattr(x, '__iter__'):
                yield from flat_iter(x)
            else:
                yield x


class LazyList(Generic[T]):
    r"""
    Lazy conversion from iterator to list. Useful when you want to support indexing into a computationally-expensive
    iterator (e.g. disk I/O). For example::

        >>> f = open('very_large_file.txt', 'r')
        >>> lazy_list = LazyList(f)
        >>> print(lazy_list[4])
        The 4th line of an extremely large file.

    :param iter_: The iterator to wrap.
    :param suffix: A fixed suffix to append to the list. When supplied, negative indexing is supported.
    """

    class _Iterator:
        def __init__(self, list_: 'LazyList'):
            self.pos = 0
            self.list = list_

        def __iter__(self):
            return self

        def __next__(self):
            item = self.list[self.pos]
            if self.list.length == -1 or self.pos < self.list.length:
                self.pos += 1
                return item
            else:
                raise StopIteration

    def __init__(self, iter_: Iterable[T], suffix: Optional[List[T]] = None):
        self.iter = iter(iter_)
        self.list: List[T] = []
        self.suffix = suffix
        self.length = -1

    def __iter__(self) -> '_Iterator':
        return self._Iterator(self)

    def __getitem__(self, item: int) -> T:
        if isinstance(item, int):
            if item < 0:
                if self.suffix is None:
                    raise ValueError("Negative indexing is only supported when `suffix` is supplied.")
                return self.suffix[item]
            try:
                while len(self.list) <= item:
                    self.list.append(next(self.iter))
            except StopIteration:
                self.length = len(self.list)
                if self.suffix is not None:
                    self.list.extend(self.suffix)
            return self.list[item]
        else:  # if isinstance(item, slice):
            raise NotImplementedError


class ListWrapper:
    r"""
    Lazy map operation for lists. Useful when you want to support indexing into :meth:`map`\ -ed
    iterator.

    .. note::
        :class:`ListWrapper` and :class:`LazyList` look similar, but have different usages:

        - :class:`ListWrapper` only takes lists, and supports advanced indexing similar to that of
          :class:`numpy.ndarray`. Mapped results are **not** cached, meaning that subsequent indexing operations
          with the same indices would result in multiple evaluations of the function.
        - :class:`LazyList` takes both lists and iterators, and only supports traditional indexing.

        The two classes can be used in combination::

            >>> f = open('very_large_file.txt', 'r')
            >>> processed_lines = ListWrapper(LazyList(f), lambda line: f'<b>{line}</b>')
            >>> print('\n'.join(processed_lines[2:4]))
            <b>The 2nd line of an extremely large file.</b>
            <b>The 3rd line of an extremely large file.</b>

    .. warning::
        This class is subject to change in the following versions.

    :param list_: The list of iterator to wrap around.
    :param func: The function to apply to iterator elements.
    """

    def __init__(self, list_, func: Callable):
        # It's hard to give ``list_`` a type: the implementation would require a list, but iterables are actually okay.
        self.func = func
        self.content = list_

    """
    Implementation Note:
    We can't just name the methods as they would be and then delete some of them, because method dispatching
    happens at the class level: unless overridden by instances, the class method will always be called.
    """

    def __iter__(self):
        return ListWrapper(iter(self.content), self.func)

    def __next__(self):
        return self.func(next(self.content))

    def __len__(self):
        return len(self.content)

    def __getitem__(self, item):
        if isinstance(item, int):
            # shortcut for the most common case
            return self.func(self.content[item])

        # multi-dimensional case
        if isinstance(item, tuple):
            index, sub = item[0], item[1:]
            if len(sub) == 1:
                sub = sub[0]
        else:
            index, sub = item, None

        if isinstance(index, slice) or hasattr(index, '__getitem__') or hasattr(index, '__iter__'):
            result = list(map(self.func, self.content[index]))
            if len(result) > 0 and isinstance(result[0], np.ndarray):
                result = np.asarray(result)
        else:
            result = self.func(self.content[index])
        if sub is not None:
            result = result[sub]
        return result


class MeasurableGenerator:
    r"""
    A wrapper around an iterator that supplies the :meth:`__len__` method. Typically used in combination with
    progress bars, e.g. :func:`progress`. For example::

        >>> lst = ['a', 'list', 'of', 5, 'elements']
        >>> for s in progress(MeasurableGenerator(map(str, lst), len(lst))):
        ...     print(s)

    :param gen: The generator.
    :param length: The length of the generator.
    """

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return MeasurableGenerator(iter(self.gen), self.length)

    def __next__(self):
        return next(self.gen)


class Range:
    r"""
    List-type over fixed range (:class:`range` with :meth:`__getitem__`). Useful when you want to support indexing into
    a :func:`range`. Usage is the same as the built-in :func:`range`::

        >>> r = Range(10)         # (end)
        >>> r = Range(1, 10 + 1)  # (start, end)
        >>> r = Range(1, 11, 2)   # (start, end, step)
        >>> print(r[0], r[2], r[4])
        1 5 9
    """

    def __init__(self, *args):
        if len(args) == 0 or len(args) > 3:
            raise ValueError("Range should be called the same way as the builtin `range`.")
        if len(args) == 1:
            self.l = 0
            self.r = args[0]
            self.step = 1
        else:
            self.l = args[0]
            self.r = args[1]
            self.step = 1 if len(args) == 2 else args[2]
        self.val = self.l
        self.length = (self.r - self.l) // self.step

    def __iter__(self):
        return Range(self.l, self.r, self.step)

    def __next__(self):
        if self.val >= self.r:
            raise StopIteration
        result = self.val
        self.val += self.step
        return result

    def __len__(self):
        return self.length

    def _get_idx(self, idx):
        return self.l + self.step * idx

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self._get_idx(idx) for idx in range(*item.indices(self.length))]
        if hasattr(item, '__getitem__') or hasattr(item, '__iter__'):
            return [self._get_idx(idx) for idx in item]
        if item < 0:
            item = self.length + item
        return self._get_idx(item)
