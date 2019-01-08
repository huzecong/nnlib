# TODO: lot's to be done here
import typing
from pathlib import Path

from .validator import ValidationError

__all__ = ['NoneType', 'Path', 'Choices']

NoneType = type(None)


class ArgType:
    def __call__(self, value):
        try:
            ret = super().__new__(value)
            ret.__init__(value)
            return ret
        except ValueError:
            raise ValidationError(f"Invalid value \"{value}\" for type ") from None


# noinspection PyUnresolvedReferences,PyProtectedMember
@typing.no_type_check
class _Choices(typing._FinalTypingBase, _root=True):
    # copied from typing._Union
    def __new__(cls, values=None, origin=None, *args, _root=False):
        self = super().__new__(cls, values, origin, *args, _root=_root)
        if origin is None:
            self.__values__ = None
            self.__args__ = None
            self.__origin__ = None
            return self
        if not isinstance(values, tuple):
            raise TypeError("Expected values=<tuple>")
        self.__values__ = values
        self.__args__ = values
        self.__origin__ = origin
        return self

    def __getitem__(self, values):
        if values == ():
            raise TypeError("Choices must contain at least one string")
        if not isinstance(values, tuple):
            values = (values,)
        values = tuple(values)
        return self.__class__(values, origin=self, _root=True)


Choices = _Choices(_root=True)
