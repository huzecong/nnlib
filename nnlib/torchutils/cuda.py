import functools
import time

from ..torch import *
from ..utils import Logging, memoization

__all__ = ['is_cuda', 'device', 'prevent_oom']


def is_cuda(module: nn.Module) -> bool:
    r"""
    Returns whether a module is on GPU, assuming that all parameters of a model are on the same device.
    """
    try:
        return next(module.parameters()).is_cuda
    except (AttributeError, TypeError):
        return False


@memoization
def device(module: nn.Module) -> torch.device:
    r"""
    Returns which device stores parameters of a module.
    """
    return next(module.parameters()).device


def save_checkpoint(model: nn.Module, optim: torch.optim.Optimizer,
                    epoch=None, train_state=None, args=None, *, filename):
    states = {
        'model_state': model.state_dict(),
        'optim': optim
    }
    # Save everything
    torch.save(states, filename)


def prevent_oom(func):
    r"""
    A function decorator that catches CUDA out of memory exceptions, triggers forced GC, and reruns the function.
    """

    import gc

    def _cuda_memory_str():
        return f"Current memory cached: {torch.cuda.memory_cached() / 1024 / 1024:.2f}MB, " \
            f"allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB"

    first_time = True

    @functools.wraps(func)
    def wrapped(model, *args, **kwargs):
        # TODO: is there a better way to inspect training loop stats?
        nonlocal first_time
        try:
            result = func(model, *args, **kwargs)
            first_time = False
            return result
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # this is an OOM error, try releasing memory and do it again
                Logging.warn(f"CUDA out of memory error caught. " + _cuda_memory_str())
                gc.collect()
                torch.cuda.empty_cache()
                Logging.warn(f"Forced GC complete. " + _cuda_memory_str())
                # now let's try again
                try:
                    result = func(model, *args, **kwargs)
                    first_time = False
                    return result
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        if first_time:
                            # OOM caused by other factors, don't bother saving
                            raise RuntimeError(f"CUDA out of memory error caught at first run. "
                                               f"Something else must be wrong.")
                        else:
                            # OOM again, this can't be fixed. Save the whole model and optimizer states
                            filename = f"{model.__class__.__name__}_{time.strftime('%Y-%m-%d %H:%M:%S')}.pt"
                            save_checkpoint(model, optim, filename=filename)
                            raise RuntimeError(f"CUDA out of memory error caught after forced GC. "
                                               f"Model & optimizer saved to {filename}. " + _cuda_memory_str())
                    else:
                        raise e  # re-raise the exception
            else:
                raise e  # re-raise the exception

    return wrapped
