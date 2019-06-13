"""
Timing Functions
"""
import contextlib
import time
from functools import wraps
from typing import Dict, Optional

from nnlib.utils.logging import Logging

__all__ = ['work_in_progress', 'tic', 'toc', 'report_timing', 'tic_toc']


@contextlib.contextmanager
def work_in_progress(msg: str):
    Logging.verbose(msg + "... ", end='')
    begin_time = time.time()
    yield
    time_consumed = time.time() - begin_time
    Logging.verbose(f"done. ({time_consumed:.2f}s)")


class _TimingHelperClass:
    ticks: float = 0
    time_records: Dict[str, float] = {}


def tic() -> None:
    _TimingHelperClass.ticks = time.time()


def toc(key: str = 'default'):
    ticks = time.time() - _TimingHelperClass.ticks

    if key not in _TimingHelperClass.time_records:
        _TimingHelperClass.time_records[key] = ticks
    else:
        _TimingHelperClass.time_records[key] += ticks


def report_timing(level: int = Logging.VERBOSE):
    Logging(level).log("Time consumed:")
    for k in sorted(_TimingHelperClass.time_records.keys()):
        v = _TimingHelperClass.time_records[k]
        Logging(level).log(f"> {k}: {v:f}")
    Logging(level).log("------")
    _TimingHelperClass.time_records = {}


def tic_toc(f):
    func_name = f.__module__ + '.' + f.__name__

    @wraps(f)
    def func(*args, **kwargs):
        tic()
        result = f(*args, **kwargs)
        toc(func_name)
        return result

    return func
