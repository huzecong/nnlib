import math
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple, TypeVar

__all__ = ['bucketed_bitext_iterator', 'sorted_bitext_iterator', 'bitext_index_iterator', 'bitext_iterator',
           'multi_dataset_iterator']

Token = TypeVar('Token', str, int)  # the token could either be a literal string or a vocabulary index
Sentence = Sequence[Token]
LangPair = Tuple[str, str]  # a pair of language specifiers


def bucketed_bitext_iterator(src_sents: Sequence[Sentence], tgt_sents: Sequence[Sentence], batch_size: int,
                             max_length: int = None) \
        -> Iterator[List[int]]:
    """
    Return an iterator generating batches over bi-text bucketed by length. Each batch only contains examples with
    source sentences of the same length.

    :param src_sents: List of source sentences.
    :param tgt_sents: List of target sentences, paired with source sentences.
    :param batch_size: Number of data examples to include in the same batch.
    :param max_length: Maximum length of both source and target sentences. If specified, examples containing sentences
        longer than this limit will be discarded.
    :return: Iterator yielding a list of indices to data examples. Each list is one batch containing no more than
        `batch_size` examples.
    """
    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx, (src_sent, tgt_sent) in enumerate(zip(src_sents, tgt_sents)):
        if max_length is not None and (len(src_sent) > max_length or len(tgt_sent) > max_length):
            continue
        if len(src_sent) == 0 or len(tgt_sent) == 0:  # skip emtpy sentences
            continue
        buckets[len(src_sent)].append(idx)

    batches = []
    src_lens = list(buckets.keys())
    random.shuffle(src_lens)
    for src_len in src_lens:
        bucket = buckets[src_len]
        random.shuffle(bucket)
        num_batches = int(math.ceil(len(bucket) * 1.0 / batch_size))
        for i in range(num_batches):
            batch = bucket[(i * batch_size):((i + 1) * batch_size)]
            batches.append(batch)
    random.shuffle(batches)
    yield from batches


def sorted_bitext_iterator(src_sents: Sequence[Sentence], tgt_sents: Sequence[Sentence], batch_size: int,
                           max_length: int = None, bins: Sequence[int] = None) \
        -> Iterator[List[int]]:
    """
    Return an iterator generating batches over bi-text. Examples in a batch are of similar length, and are sorted in
    descending order of source sentence length.

    In implementation, examples are first clustered into bins (as in `np.histogram`).

    :param src_sents: List of source sentences.
    :param tgt_sents: List of target sentences, paired with source sentences.
    :param batch_size: Number of data examples to include in the same batch.
    :param max_length: Maximum length of both source and target sentences. If specified, examples containing sentences
        longer than this limit will be discarded.
    :param bins: Thresholds for bins.
    :return: Iterator yielding a list of indices to data examples. Each list is one batch containing no more than
        `batch_size` examples.
    """
    bins = bins or [20, 30, 40, 50, 60, 75]
    grouped_data: List[List[int]] = [[] for _ in bins]

    outlier = []
    for idx, (src_sent, tgt_sent) in enumerate(zip(src_sents, tgt_sents)):
        if max_length is not None and (len(src_sent) > max_length or len(tgt_sent) > max_length):
            continue
        if len(src_sent) == 0 or len(tgt_sent) == 0:  # skip emtpy sentences
            continue
        for bid, i in enumerate(bins):
            if len(src_sent) <= i:
                grouped_data[bid].append(idx)
                break
        else:
            outlier.append(idx)
    if len(outlier) > 0:
        grouped_data.append(outlier)

    batches = []
    for group in grouped_data:
        random.shuffle(group)
        num_batches = int(math.ceil(len(group) * 1.0 / batch_size))
        for i in range(num_batches):
            batch = group[(i * batch_size):((i + 1) * batch_size)]
            batch = sorted(batch, key=lambda x: len(src_sents[x]), reverse=True)
            batches.append(batch)
    random.shuffle(batches)
    yield from batches


def bitext_index_iterator(src_sents: Sequence[Sentence], tgt_sents: Sequence[Sentence], batch_size: int,
                          max_length: int = None, sort: bool = False) \
        -> Iterator[List[int]]:
    """
    A convenient interface that calls other bi-text iterator functions. Returns an iterator over example indices.

    :param src_sents: List of source sentences.
    :param tgt_sents: List of target sentences, paired with source sentences.
    :param batch_size: Number of data examples to include in the same batch.
    :param max_length: Maximum length of both source and target sentences. If specified, examples containing sentences
        longer than this limit will be discarded.
    :param sort: If true, the examples in a batch are sorted in descending order of source sentence length.
    """
    if sort:
        iterator = sorted_bitext_iterator(src_sents, tgt_sents, batch_size, max_length=max_length)
    else:
        iterator = bucketed_bitext_iterator(src_sents, tgt_sents, batch_size, max_length=max_length)
    yield from iterator


def bitext_iterator(src_sents: Sequence[Sentence], tgt_sents: Sequence[Sentence], batch_size: int,
                    max_length: int = None, sort: bool = False) \
        -> Iterator[Tuple[List[Sentence], List[Sentence]]]:
    """
    A wrapper over :method:`bitext_index_iterator` that returns the actual data examples instead of indices.
    """
    iterator = bitext_index_iterator(src_sents, tgt_sents, batch_size, max_length=max_length, sort=sort)
    for batch in iterator:
        src_batch = [src_sents[idx] for idx in batch]
        tgt_batch = [tgt_sents[idx] for idx in batch]
        yield src_batch, tgt_batch


Key = TypeVar('Key')
BatchValue = TypeVar('BatchValue')


def multi_dataset_iterator(iterators: Mapping[Key, Iterator[BatchValue]]) \
        -> Iterator[Tuple[Key, BatchValue]]:
    """
    An iterator that first chooses a dataset at random, then returns a batch from that dataset.

    :param iterators: A dictionary mapping keys to iterators.
    :return: Iterator yielding the key of the chosen dataset and the batch.
    """
    iters_list = list(iterators.items())  # create a list for indexing
    while len(iters_list) > 0:
        iter_idx = random.randint(0, len(iters_list) - 1)
        key, it = iters_list[iter_idx]
        try:
            batch = next(it)
        except StopIteration:
            del iters_list[iter_idx]
            continue
        yield key, batch
