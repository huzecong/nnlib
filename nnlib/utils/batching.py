from typing import List, Sequence, Tuple, TypeVar, overload

import numpy as np

from .iterable import MeasurableGenerator
from .math import ceil_div
from ..torch import *

__all__ = ['minibatches_from', 'pad_sequences', 'batch_sequences', 'shift_packed_seq', 'mask_dropout_embeddings']


def _minibatches_from(data, size=16, shuffle=True, different_size=False):
    length = len(data)
    if shuffle:
        idxs = np.random.permutation(ceil_div(length, size)) * size
    else:
        idxs = np.arange(ceil_div(length, size)) * size
    for idx in idxs:
        if not different_size and idx + size > length:
            batch = list(data[-size:])
        else:
            batch = list(data[idx:(idx + size)])
        yield batch


def minibatches_from(data, size=16, shuffle=True, different_size=False):
    r"""
    A low-level API to directly create mini-batches from a list.

    :param data: Data to create mini-batches for.
    :param size: Batch size.
    :param shuffle: Whether to shuffle data.
    :param different_size: If ``True``, allows the final batch to have different size than specified batch size.
        If ``False``, extra elements from end of array will be appended.
    :return: A generator yielding one batch at a time.
    """
    generator = _minibatches_from(data, size, shuffle, different_size)
    length = ceil_div(len(data), size)
    return MeasurableGenerator(generator, length)


T = TypeVar('T')


def pad_sequences(seqs: List[List[int]], batch_first=False, pad: int = -1) -> LongTensor:
    r"""
    A wrapper around :func:`nn.utils.rnn.pad_sequence` that takes a list of lists, and converts it into a list of
    :class:`torch.LongTensor`\ s.
    """
    tensor_seqs = [torch.tensor(seq, dtype=torch.long) for seq in seqs]
    return nn.utils.rnn.pad_sequence(tensor_seqs, batch_first=batch_first, padding_value=pad)


@overload
def batch_sequences(seqs: Sequence[Sequence[T]], ordered=False) -> PackedSequence: ...


@overload
def batch_sequences(seqs: Sequence[Sequence[T]], *args: Sequence, ordered=False) \
        -> Tuple[PackedSequence, Tuple[Sequence, ...]]: ...


def batch_sequences(seqs, *args, ordered=False):
    r"""
    Given a batch from data loader, convert it into PyTorch :class:`PackedSequence`.

    Since :class:`PackedSequence` requires sequences to be sorted in reverse order of their length, batch elements are
    reordered. If batch contains other data apart from the sequence (e.g. labels), pass them in ``*args`` to allow
    consistent reordering.

    :param seqs: The sequences.
    :param args: Other data to be reordered.
    :param ordered: If true, assume (and check) that the sequences are already sorted in decreasing length.
    :param pack: If true, return a PackedSequence, otherwise return a Tensor.
    """
    indices = sorted(range(len(seqs)), key=lambda idx: len(seqs[idx]), reverse=True)
    if ordered and indices != list(range(len(seqs))):
        raise ValueError("Sentences are not sorted in decreasing length while specifying `ordered=True`")
    tensor_seqs = [torch.tensor(seqs[idx], dtype=torch.long) for idx in indices]
    reordered_args = tuple([xs[idx] for idx in indices] for xs in args)
    packed_seq: PackedSequence = nn.utils.rnn.pack_padded_sequence(
        nn.utils.rnn.pad_sequence(tensor_seqs, padding_value=-1), [len(seq) for seq in tensor_seqs])
    if len(reordered_args) > 0:
        return packed_seq, reordered_args
    return packed_seq


def shift_packed_seq(seq: PackedSequence, start: int = 0) -> PackedSequence:
    r"""
    Shifts the :class:`PackedSequence`, i.e. return the substring starting from ``start``.

    :param seq: The sequence to truncate.
    :param start: The left endpoint of truncate range. Supports Python indexing syntax (negative for counting
                  from the end).
    :return: Truncated sequence.
    """
    data, batch_sizes = seq
    return PackedSequence(data[sum(batch_sizes[:start]):], batch_sizes[start:])


def mask_dropout_embeddings(strs: Sequence[Sequence[T]], dropout_prob: float, transposed=False) -> np.ndarray:
    r"""
    Generate mask for embedding dropout, each word type is either dropped out as a whole
    or scaled according to the dropout probability.

    :param strs: Un-transposed sentence batch.
    :param dropout_prob: Token dropout probability.
    :param transposed: Whether to return transposed mask (i.e. batch_size * length)
    :return: The generated mask.
    """
    words = set([w for s in strs for w in s])
    dropout = dict(zip(words, np.random.binomial(1, 1 - dropout_prob, len(words))))
    if transposed:
        mask = np.full((len(strs), len(strs[0])), 1 / (1 - dropout_prob), dtype=np.bool)
        for i in range(len(strs)):
            for j, w in enumerate(strs[i]):
                mask[i, j] = dropout[w]
    else:
        max_len = (max(len(s) for s in strs))
        mask = np.full((max_len, len(strs)), 1 / (1 - dropout_prob), dtype=np.bool)
        for idx, s in enumerate(strs):
            mask[:len(s), idx] = [dropout[w] for w in s]
    # inverse dropout
    return mask
