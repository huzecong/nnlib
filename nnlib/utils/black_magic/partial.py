import types
import functools

""" WIP """

raise NotImplementedError


def partial(*default_args, **default_kwargs):
    r"""
    The builtin `functools.partial` is slow; use this instead.
    NOTE: Efficiency is achieved by sacrificing features, this method does not support:
        - Extendable variadic arguments (*args). If the given function contains `*args`, and you supplied more
            arguments than the given function supports, those arguments will be used as `*args`, and you will
            NOT be able to add more arguments to it.
        - Unspecified keyword arguments (**kwargs). Same as above.
        - Built-in functions/methods. When given built-in functions or methods, `functools.partial` will be called.
    """
    import inspect

    def decorator(f):
        if hasattr(f, '__closure__'):
            # Function or static method
            is_method = False
            func = types.FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
        elif hasattr(f, '__func__'):
            # Instance method
            is_method = True
            mf = f.__func__
            func = types.FunctionType(mf.__code__, mf.__globals__, mf.__name__, mf.__defaults__, mf.__closure__)
        else:
            # Built-in function/method
            return functools.partial(f, *default_args, **default_kwargs)

        argspec = inspect.getargspec(func)

        if is_method:
            method = types.MethodType(func, f.__self__, f.__self__.__class__)
            return method
        else:
            return func
