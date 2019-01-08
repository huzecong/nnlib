Argument Parsing - ``nnlib.arguments``
======================================

.. automodule:: nnlib.arguments
.. currentmodule:: nnlib.arguments


The ``arguments`` package provides a replacement for the built-in argument parser :mod:`argparse`.


The :class:`Arguments` Class
----------------------------

Yet Another command line argument class, but with IDE support.

The :class:`Arguments` class is designed with the goals in mind (in decreasing order of importance):

- Index-able by IDEs and IPython.
- Non-verbose, intuitive syntax.
- Support custom verification.
- Valid replacement for :mod:`argparse` for simple usages.

Using :class:`Arguments` is simple. Simply subclass and add attributes::

    class MyArguments(Arguments):
        batch_size: int = 32
        optimizer: Choice('sgd', 'adam', 'adagrad') = 'sgd'
        cuda = Arguments.Switch()
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

.. note::
	The following argument is provided by default:

	- ``ipdb`` (:class:`Switch`): If on, spawn an IPython debugger on exceptions. Requires having the package ``ipdb`` installed.


Type Resolution
---------------

Python annotations are used as both type specification and arguments options. When not specified, type will be deduced from the default value. In this case, the default value cannot be ``None``.

Note that it is difficult to perform complex type checks at run time (especially for custom and generic types). Instead of writing lengthy and hacky code for such pointless functionality, we use the following rule-based method:

- If argument type is :class:`typing.Optional`, allow value to be `None`. If not `None`, use parameter type and follow the rest of rules.

- If argument type is a simple Python built-in type (:class:`bool`, :class:`int`, :class:`float`, :class:`str`), perform type coercion.

    - Specifically for bool, command-line syntax could be switch-style (`--flag`, only if default value is `False`), or value-style using anything you find reasonable (`--flag true`, `--flag Y`, etc).

- If argument type is a container generic (:class:`typing.List`, :class:`typing.Tuple`, :class:`Set`), or their corresponding types (:class:`list`, :class:`tuple`, :class:`set`), treat value as list and then perform coercion.

    - Lists could be specified using Python syntax (``[1,2,3]``, remember not to put spaces unless wrapped in quotes), or a comma-separated string (``1,2,3``). Only the former syntax supports complex lists e.g. nested lists.
    - Each list element is recursively checked against the container parameter type (if specified using :mod:`typing` module).

- If argument type is :class:`typing.Dict` or :class:`dict`, treat value as dict.

    - Python syntax must be used (``{'a':1,'b':2}``). Key and value types are checked recursively (if specified).

- If argument type is :class:`~validator.Choice`, value is compared to string forms of each choice and the matched choice is used. If no match is found, a :class:`ValueError` is raised.

    - :class:`~validator.Choice` is inherited from :class:`typing.Union`, so IDEs should be able to recognize.
    - However, it is not recommended to mix choices of different types. If `False` or `None` should be one of the choices, either include it as literal ``none`` or use :class:`~typing.Optional`.

- If argument type is :class:`~validator.Path`, value is stored as :class:`str` but checked to be of correct format. If `exists=True`, then :meth:`os.path.exists` is called for validation.

- If argument type is any other type, function, or callable object, it is called with the value (:class:`str` form). This is not good practice and should be avoided.

.. note::
    The above is the intended behavior, but is not quite what is implemented as of now.

    Currently, if the argument type is not among built-in types, :class:`~custom_types.Choice`, and
    :class:`~validator.Path`, we simply fallback to :func:`ast.literal_eval` and do not perform type checking.


API
---

.. autoclass:: Arguments
	:members:

.. autoclass:: nnlib.arguments.custom_types.Choices


Validators
----------

.. currentmodule:: nnlib.arguments.validator

.. autofunction:: is_path
.. autofunction:: is_embedding_type
.. autofunction:: is_dropout
.. autofunction:: is_lstm_dropout
.. autofunction:: is_activation
.. autofunction:: is_optimizer
