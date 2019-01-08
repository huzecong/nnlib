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
