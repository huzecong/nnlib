import functools


class DynamicValue:
    """
    **Warning:** Black magic, use with caution!!

    Treat a function as the value it returns, for example::

      x = 2; a = DynamicValue(lambda: x)
      print 3 * a + 2  # 8
      x = 4
      print 3 * a + 2  # 14

    Since Python adopts duck typing, a value could be replaced by a function that returns values in specific type.
    This is useful when you need *dynamic* values that will be fed into some black-box function (given the value is
    not copied).

    This is done by evaluating the function when building the :py:class:`DynamicValue` instance, and re-route all
    methods calls to the value's method. This is called method dispatching, and is implemented at both instance-level
    and also class-level, because Python defaults instance-level methods to its class-level ones.
    """

    def _dispatch(self, name, method):
        @functools.wraps(method, assigned=['__name__', '__doc__'])
        def func(*args, **kwargs):
            return getattr(self._func(), name)(*args, **kwargs)

        return func

    @staticmethod
    def _class_dispatch(name, method):
        @functools.wraps(method, assigned=['__name__', '__doc__'])
        def func(self, *args, **kwargs):
            return getattr(self, name)(*args, **kwargs)

        return func

    def __init__(self, func):
        self._func = func
        value = func()
        for k in dir(value):
            if k in ['__class__', '__getattr__', '__getattribute__', '__setattr__', '__new__', '__init__']:
                continue
            v = getattr(value, k)
            if hasattr(v, '__call__'):
                # Forcing early binding of for-loop variable using `_dispatch` and `_class_dispatch` factories
                func = self._dispatch(k, v)
                setattr(self, k, func)
                class_func = DynamicValue._class_dispatch(k, v)
                setattr(DynamicValue, k, class_func)

    def value(self):
        return self._func()

    def __getattr__(self, item):
        return getattr(self._func(), item)
