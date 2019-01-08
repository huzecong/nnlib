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
    import torch.nn.functional as F
    import torch.nn.utils
    # noinspection PyUnresolvedReferences
    from torch import Tensor, LongTensor
    from torch.autograd import Variable
    from torch.nn.utils.rnn import PackedSequence

STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = False  # global['STATICA_HACK'] = False
if STATICA_HACK:
    # this never runs. didn't expect that did you PyCharm?
    from . import workaround as torch
    from .workaround import Tensor, LongTensor

HiddenState = Tuple[Tensor, Tensor]

__all__ = ['torch', 'nn', 'F', 'Tensor', 'Variable', 'LongTensor',
           'PackedSequence', 'HiddenState']
