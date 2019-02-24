import sys


class _PersistentLocals(object):
    def __init__(self, func):
        self._locals = {}
        self.func = func

    def __call__(self, *args, **kwargs):
        # noinspection PyUnusedLocal
        def tracer(frame, event, arg):
            if event == 'return':
                self._locals = frame.f_locals

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            # trace the function call
            res = self.func(*args, **kwargs)
        finally:
            # disable tracer and replace with old one
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}

    @property
    def locals(self):
        return self._locals


# noinspection PyPep8Naming
def IPython_main(confirm=False):
    r"""
    When executing code in IPython using the ``%run`` magic, global variables are exposed to the REPL environment when
    code finishes, or an exception is raised. This makes debugging or stage-by-stage execution easy because you can
    manipulate variables afterwards.

    However, this may not be sufficient under certain circumstances. A good paradigm is to keep everything in a
    ``main()`` function, instead of under a ``if __name__ == '__main__'`` statement, because this keeps variables local
    so they don't clash with other stuff (and the IDE doesn't warn you about shadowing names). But in this case, IPython
    will not expose the local variables.

    This decorator is used to expose local variables of a function into the global namespace upon exiting or raising
    exceptions. It only functions when it is run inside IPython. You can use the decorator as follows:

    .. code-block:: python

        @IPython_main  # equivalent to @IPython_main()
        def main(): ...

        @IPython_main(confirm=True)
        def main(): ...

    .. warning::
        This decorator is implemented via a custom profiler, which is called every time a function is called or is
        about to return. The overhead is not tested, but it is likely to slow things down. Avoid when possible.
    """
    import functools
    import typing

    def decorator(main: typing.Callable[[], None]):
        # Check if we're in an IPython shell
        try:
            # noinspection PyUnresolvedReferences
            get_ipython  # type: ignore
        except NameError:
            # Not in IPython interactive shell, keep function as is
            return main
        else:
            @functools.wraps(main)
            def wrapped(*args, **kwargs):
                # Wrap it using some black magic
                main_wrapped = _PersistentLocals(main)
                try:
                    # Run the main function as usual
                    main_wrapped(*args, **kwargs)
                except Exception:
                    raise
                finally:
                    # Magic happens
                    if confirm:
                        print('', file=sys.stderr)
                        print(list(main_wrapped.locals.keys()), file=sys.stderr)
                        raw_input = input(
                            'The above variables are going to be exposed to IPython console, confirm ? (y/n) ')
                        while True:
                            if raw_input.lower() in {'yes', 'y', 'ok', 't', 'true', '1'}:
                                choice = True
                            elif raw_input.lower() in {'no', 'n', 'f', 'false', '0'}:
                                choice = False
                            else:
                                raw_input = input('Please answer "y" or "n": ')
                                continue
                            break
                        if not choice:
                            return
                    main_wrapped.func.__globals__.update(main_wrapped.locals, )
                    print('😈 Black magic successfully cast', file=sys.stderr)

            return wrapped

    if type(confirm) is not bool:  # also allow using `@IPython_main` without arguments
        main_func = confirm
        confirm = False
        # noinspection PyTypeChecker
        return decorator(main_func)
    return decorator
