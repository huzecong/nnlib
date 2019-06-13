import os
from typing import Tuple

if 'READTHEDOCS' in os.environ:
    class _Dummy:
        def __getattr__(self, item):
            if ord('A') <= ord(item[0]) <= ord('Z'):
                return _Dummy
            else:
                return _Dummy()

    # since torch is not installed on readthedocs, we just mock them
    for name in ['torch', 'nn', 'F', 'utils']:
        globals()[name] = _Dummy()
    for typ in ['Tensor', 'LongTensor', 'Variable', 'PackedSequence']:
        globals()[typ] = _Dummy
else:
    import torch
    import torch.nn as nn
    # noinspection PyPep8Naming
    import torch.nn.functional as F
    import torch.nn.utils
    # noinspection PyUnresolvedReferences
    from torch import Tensor, LongTensor
    from torch.autograd import Variable
    from torch.nn.utils.rnn import PackedSequence

HiddenState = Tuple[Tensor, Tensor]

__all__ = ['torch', 'nn', 'F', 'Tensor', 'Variable', 'LongTensor',
           'PackedSequence', 'HiddenState']
