import ast
import enum
import re
import sys
import types
import typing
from pathlib import Path

from nnlib.utils import Logging
from . import custom_types
from .validator import ValidationError


class Arguments:
    """
    Yet Another command line argument class, but with IDE support.

    The :class:`Arguments` class is designed with the goals in mind (in decreasing order of importance):

    - Index-able by IDEs and IPython.
    - Non-verbose, intuitive syntax.
    - Support custom verification.
    - Valid replacement for :module:`argparse` for simple usages.

    Using :class:`Arguments` is simple. Simply subclass and add attributes::

        class MyArguments(Arguments):
            batch_size: int = 32
            optimizer: Choice('sgd', 'adam', 'adagrad') = 'sgd'
            cuda = False
            languages: Optional[List[str]] = None
            data_path: Path(exists=True) = None

    These arguments can be specified through command line::

        python main.py \
            --batch-size 16 \
            --optimizer adam \
            --cuda \
            --languages ['de','en','it']  # no spaces!

    or programmatically::

        args = MyArguments(languages=['de', 'en'])

    Python annotations are used as both type specification and arguments options. When not specified, type will be
    deduced from the default value. In this case, the default value cannot be ``None``.

    Note that it is difficult to perform complex type checks at run time (especially for custom and generic types).
    Instead of writing lengthy and hacky code for such pointless functionality, we use the following rule-based method:

    - If argument type is :class:`typing.Optional`, allow value to be `None`. If not `None`, use parameter type and
      follow the rest of rules.

    - If argument type is a simple Python built-in type (:class:`bool`, :class:`int`, :class:`float`, :class:`str`),
      perform type coercion.
        - Specifically for bool, command-line syntax could be switch-style (`--flag`, only if default value is `False`),
          or value-style using anything you find reasonable (`--flag true`, `--flag Y`, etc).

    - If argument type is a container generic (:class:`typing.List`, :class:`typing.Tuple`, :class:`Set`), or their
      corresponding types (:class:`list`, :class:`tuple`, :class:`set`), treat value as list and then perform coercion.
        - Lists could be specified using Python syntax (`[1,2,3]`, remember not to put spaces unless wrapped in quotes),
          or a comma-separated string (`1,2,3`). Only the former syntax supports complex lists e.g. nested lists.
        - Each list element is recursively checked against the container parameter type (if specified using
          :module:`typing` module).

    - If argument type is :class:`typing.Dict` or :class:`dict`, treat value as dict.
        - Python syntax must be used (`{'a':1,'b':2}`). Key and value types are checked recursively (if specified).

    - If argument type is :class:`~validator.Choice`, value is compared to string forms of each choice and the matched
      choice is used. If no match is found, a :class:`ValueError` is raised.
        - :class:`~validator.Choice` is inherited from :class:`typing.Union`, so IDEs should be able to recognize.
        - However, it is not recommended to mix choices of different types. If `False` or `None` should be one of the
          choices, either include it as literal `"none"` or use :class:`~typing.Optional`.

    - If argument type is :class:`~validator.Path`, value is stored as :class:`str` but checked to be of correct format.
      If `exists=True`, then :meth:`os.path.exists` is called for validation.

    - If argument type is any other type, function, or callable object, it is called with the value (:class:`str` form).
      This is not good practice and should be avoided.
    """

    _reserved_keys = ['Switch', 'validate', 'Enum']

    class Enum(enum.Enum):
        def _generate_next_value_(name, start, count, last_values):
            return name

        def __eq__(self, other):
            return self.value == other or super().__eq__(other)

    class Switch:
        def __init__(self):
            self._value = False

        def __bool__(self):
            return self._value

        def __repr__(self):
            return f"(Switch) {self._value}"

    debug = Switch()
    ipdb = Switch()

    def _check_types(self):
        pass

    def __init__(self, **kwargs):
        self._check_types()

        for k, v in kwargs.items():
            setattr(self, k, v)

        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg.startswith('--'):
                argname = arg[2:].replace('-', '_')
                if hasattr(self, argname):
                    attr = getattr(self, argname)
                    if isinstance(attr, Arguments.Switch):
                        attr._value = True
                        i += 1
                        continue

                    typ = self.__annotations__.get(argname, type(attr))
                    nullable = False
                    # TODO: hacks here
                    if hasattr(typ, '__origin__') and typ.__origin__ == typing.Union and type(None) in typ.__args__:
                        # hacky check of whether `typ` is `Optional`
                        nullable = True
                        typ = next(t for t in typ.__args__ if not isinstance(t, custom_types.NoneType))  # type: ignore
                    argval: str = sys.argv[i + 1]
                    if argval.lower() == 'none':
                        assert nullable, f"Argument '{argname}' is not nullable"
                        val = None
                    elif isinstance(typ, custom_types.NoneType):  # type: ignore
                        val = None  # just to suppress "ref before assign" warning
                        try:
                            # priority: low -> high
                            for target_typ in [str, float, int]:
                                val = target_typ(argval)
                        except ValueError:
                            pass
                    elif typ is str:
                        val = argval
                    elif isinstance(typ, custom_types.Path) or typ is custom_types.Path:
                        val = Path(argval)
                        if isinstance(typ, custom_types.Path) and typ.exists:
                            assert val.exists(), ValueError(f"Argument '{argname}' requires an existing path, "
                                                            f"but '{argval}' does not exist")
                    elif isinstance(typ, custom_types._Choices):
                        val = argval
                        assert val in typ.__values__, f"Invalid value '{val}' for argument '{arg}', " \
                                                      f"available choices are: {typ.__values__}"
                    elif issubclass(typ, Arguments.Enum):
                        # experimental support for custom enum
                        try:
                            # noinspection PyCallingNonCallable
                            val = typ(argval)
                        except ValueError:
                            valid_args = {x.value for x in typ}
                            raise ValueError(f"Invalid value '{argval}' for argument '{argname}', "
                                             f"available choices are: {valid_args}") from None

                    elif typ is bool:
                        val = argval in ['true', '1', 'True', 'y', 'yes']
                    else:
                        try:
                            val = ast.literal_eval(argval)
                        except ValueError:
                            raise ValueError(f"Invalid value '{argval}' for argument '{argname}'") from None
                    setattr(self, argname, val)
                    i += 2
                else:
                    raise ValueError(f"Invalid argument: '{arg}'")
            else:
                Logging.warn(f"Unrecognized command line argument: '{arg}'")
                i += 1

        if self.ipdb:
            # enter IPython debugger on exception
            from IPython.core import ultratb
            sys.excepthook = ultratb.FormattedTB(mode='Context', color_scheme='Linux', call_pdb=1)

        self._validate()

    def _validate(self):
        rules = self.validate() or []
        for pattern, validator in rules:
            regex = re.compile(rf"^{pattern}$")
            for k in dir(self):
                if not k.startswith('_') and regex.match(k):
                    v = getattr(self, k)
                    try:
                        result = validator(v)
                    except Exception:
                        raise ValidationError(k, v, validator.__name__)
                    else:
                        if not result:
                            raise ValidationError(k, v, validator.__name__)

    def validate(self):
        """
        Return a list of validation rules. Each validation rule is a tuple of (pattern, validator), where:

        - ``pattern``: A regular expression string. The validation rule is applied to all arguments whose name is
          fully-matched (i.e. ``^$``) by the pattern.
        - ``validator``: A validator instance from ``arguments.validator``.

        Example::

            def validate(self):
                return [
                    ('.*_data_path', validator.is_path()),
                    ('pretrained_embedding_type', validator.is_embedding_type()),
                    ('pretrained_embedding_path', validator.is_path(nullable=True)),
                    ('.*_lstm_dropout', validator.is_dropout()),
                ]

        :return: List of (pattern, validator).
        """
        pass

    @staticmethod
    def _dispatch(v, func):
        return lambda: v.value(func())

    def __str__(self):
        s = repr(self.__class__) + '\n' + '-' * 75 + '\n'
        max_key = max(len(k) for k in dir(self) if not k.startswith('_') and k not in self._reserved_keys)
        for k in sorted(dir(self)):
            if k.startswith('_') or k in self._reserved_keys:
                continue
            v = getattr(self, k)
            if isinstance(type(v), types.MethodType):
                continue
            s += f"{k}{' ' * (2 + max_key - len(k))}{v!r}\n"
        s += ('-' * 75)
        return s
