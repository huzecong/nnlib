from ..torch import *

__all__ = ['release_cuda_memory']


def release_cuda_memory(trainer):
    torch.cuda.empty_cache()
