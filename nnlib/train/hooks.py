from nnlib.torch import *
from nnlib.train.trainer import Trainer

__all__ = ['release_cuda_memory']


def release_cuda_memory(trainer: Trainer):
    torch.cuda.empty_cache()
