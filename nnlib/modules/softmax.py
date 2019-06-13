import operator
from typing import Optional, Sequence, Mapping, Union

import numpy as np

from nnlib import utils
from nnlib.modules.linear import Linear
from nnlib.torch import *

__all__ = ['Softmax', 'AdaptedSoftmax']


class Softmax(nn.Module):
    r"""
    Softmax for classification. You know what I mean.

    Follows the same interface as :py:class:`AdaptedSoftmax`.
    """

    def __init__(self, vocab_size: int, embed_dim: int, bias=False, *, _weight: Optional[Tensor] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = Linear(in_features=embed_dim, out_features=vocab_size, bias=bias, _weight=_weight)

    def forward(self,  # type: ignore
                input: Tensor, target: torch.LongTensor, reduction='elementwise_mean',
                ignore_index: Optional[int] = None, smoothing: float = 0.0) -> Tensor:
        r"""
        :param input: Input of shape ``batch_size * ... * embed_dim``.
        :param target: LongTensor of shape ``batch_size * ...``, corresponding to indices. Each element should be in
            range ``[0, vocab_size)``.
        :param reduction: Reduction setting as in :class:`nn.NLLLoss`.
        :param ignore_index: If specified, targets with this index is ignored and not used in reduction.
        :param smoothing: Label smoothing weight. The target distribution becomes:

            .. math::

                P(x) = \begin{cases}
                    1 - \mathrm{smoothing}                          & x = \mathrm{target} \\
                    \mathrm{smoothing} / (\mathrm{vocab\_size} - 1) & x \neq \mathrm{target} \\
                \end{cases}

        :return: Negative log-likelihood.
        """
        if target.device != self.linear.weight.device:
            target = target.to(device=self.linear.weight.device)
        logits = self.linear.forward(input)
        probs = F.log_softmax(logits, dim=1)
        if smoothing == 0.0:
            loss = F.nll_loss(probs, target, reduction=reduction, ignore_index=(ignore_index or -100))
        else:
            # TODO: Add support for ignore_index
            smoothed_prob = smoothing / (self.vocab_size - 1)
            loss = -torch.sum(probs, dim=1) * smoothed_prob
            loss -= (torch.gather(probs, dim=1, index=target.contiguous().view(-1, 1)).view(-1) *
                     (1 - smoothing - smoothed_prob))
            if reduction == 'elementwise_mean':
                loss = torch.mean(loss)
            elif reduction == 'sum':
                loss = torch.sum(loss)
            elif reduction == 'none':
                loss = loss
            else:
                raise ValueError(f"Invalid reduction method \"{reduction}\"")
        return loss

    def log_prob(self, input: Tensor) -> Tensor:
        logits = self.linear.forward(input)
        return torch.log_softmax(logits, dim=1)

    def predict(self, input: Tensor) -> LongTensor:
        logits = self.linear(input)
        return torch.argmax(logits, dim=1)


class AdaptedSoftmax(nn.AdaptiveLogSoftmaxWithLoss):
    r"""
    Wrapper over :py:class:`torch.nn.AdaptiveLogSoftmaxWithLoss`. Maintains a mapping of data indices to softmax
    indices based on frequency.
    """

    indices: LongTensor
    reverse_indices: LongTensor

    def __init__(self, vocab_size: int, embed_dim: int, vocab_freq: Union[Mapping[int, int], Sequence[int]],
                 cutoffs: Optional[Sequence[int]] = None, **kwargs):
        r"""
        :type vocab_freq: list of int
        :type cutoffs: None | list of int

        :param vocab_freq: List of frequencies for each word.
        :param cutoffs: Manually specify cluster partitions, or ``None`` to use default partition scheme.
        :param kwargs: Remaining keyword arguments passed to :py:class:`torch.nn.AdaptiveLogSoftmaxWithLoss`
        """

        if cutoffs is None:
            probs = np.asarray(vocab_freq)
            probs /= np.sum(probs)
            prefix_prob = utils.scanl(operator.add, sorted(probs))
            cutoffs = [next(vocab_size - k for k, p in enumerate(prefix_prob) if 1.0 - p < cut_p)
                       for cut_p in [0.8, 0.9, 0.95]]

        super().__init__(embed_dim, vocab_size, cutoffs=cutoffs, **kwargs)

        indices = torch.tensor(sorted(range(vocab_size), key=lambda x: vocab_freq[x], reverse=True))
        reverse_indices = torch.tensor(utils.reverse_map(self.indices), dtype=torch.long)
        self.register_buffer('indices', indices)
        self.register_buffer('reverse_indices', reverse_indices)

    def forward(self,  # type: ignore
                input: Tensor, target: torch.LongTensor, reduction: str = 'elementwise_mean') -> Tensor:
        log_likelihood, _loss = super().forward(input, self.indices[target])
        if reduction == 'elementwise_mean':
            loss = _loss
        elif reduction == 'sum':
            loss = torch.sum(-log_likelihood)
        elif reduction == 'none':
            loss = -log_likelihood
        else:
            raise ValueError(f"Invalid reduction method \"{reduction}\"")
        return loss

    def log_prob(self, input: Tensor) -> Tensor:
        output = super().log_prob(input)
        mapped = torch.index_select(output, dim=1, index=self.reverse_indices)
        return mapped

    def predict(self, input: Tensor) -> LongTensor:
        output = super().predict(input)
        return self.reverse_indices[output]
