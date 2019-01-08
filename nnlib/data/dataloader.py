import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, \
    TypeVar, Union, overload

import numpy as np

from .. import utils
from ..utils import Logging, path_add_suffix

__all__ = ['Vocabulary', 'DataLoader', 'read_file', 'read_bitext_files']

K = TypeVar('K')
V = TypeVar('V')


class Vocabulary(defaultdict, Generic[K, V]):
    @classmethod
    def nop(cls):
        pass

    def __init__(self, default_factory: Optional[Callable[[], V]] = None, load=True, **kwargs):
        if default_factory is None:
            super().__init__(self.__len__)
        elif default_factory is Vocabulary.nop:
            super().__init__()
        else:
            super().__init__(default_factory)
        self._is_frozen = False
        self.i2w: Dict[V, K] = {}
        self.info: Dict[str, V] = {}
        self._load = load
        self._loaded = False
        for k, v in kwargs.items():
            token = self[v]
            setattr(self, k, token)
            self.info[k] = token

    def __missing__(self, key: K) -> V:
        if self._is_frozen:
            raise ValueError(f"Trying to modify a frozen vocabulary with key '{key}'")
        value = super().__missing__(key)
        self.i2w[value] = key
        return value

    def __setitem__(self, key: K, value: V):
        if self._is_frozen:
            raise ValueError(f"Trying to modify a frozen vocabulary with key '{key}' and value '{value}'.")
        super().__setitem__(key, value)

    def update(self, m: Union[K, Sequence[K]], **kwargs):
        """"""
        if isinstance(m, dict):
            super().update(m)
        elif isinstance(m, list):
            _ = [self[k] for k in m]
        else:
            _ = self[m]

    def to_word_list(self, sent: Sequence[V]) -> List[K]:
        return [self.i2w[tok] for tok in sent]

    def to_str(self, sent: Sequence[V]) -> str:
        return ' '.join(map(str, self.to_word_list(sent)))

    @classmethod
    def load(cls, path: Path) -> 'Vocabulary':
        with path.open('rb') as f:
            vocab = pickle.load(f)
            vocab_info: dict = pickle.load(f)
        vocab.info = vocab_info
        for k, v in vocab_info.items():
            setattr(vocab, k, v)
        vocab.i2w = {v: k for k, v in vocab.items()}
        return vocab

    def save(self, path: Path):
        tmp_factory = self.default_factory
        self.default_factory = None
        with path.open('wb') as f:
            pickle.dump(self, f)
            pickle.dump(self.info, f)
        self.default_factory = tmp_factory

    def freeze(self) -> None:
        self._is_frozen = True


Token = TypeVar('Token')


class DataLoader(Generic[Token]):
    """
    An abstract data loader interface for neural networks, utilities included:

    - :py:class:`Vocabulary` for automatic token indexing and mapping.
    - Vocabulary pruning based on frequency.
    - Byte-pair encoding.
    - Batching, shuffling (w/ or w/o buffer), and binning.

    You should subclass :py:class:`DataLoader` and implement the following methods:

    - ``__init__``: Define vocabularies and stuff.
    - ``__len__``: Return dataset size.
    - ``iterdata``: Iterate over examples one by one.
    - ``generate_vocab``: Generate vocabularies. This is called on the first run or when vocabularies
      need to be updated.
    - ``preprocess``: Preprocessing stuff after reading vocabularies. This is where you add SOS/EOS/UNKs
      and load data into memory (if required).

    Also, remember to change ``__version__`` when you make modifications to the data loader subclass, since this will
    allow automatic rebuilding of vocabularies. ``__version__`` can be any string, and a nice convention is to use the
    date plus version, e.g. ``20181003v2``.
    """

    # When making updates to the data loader, also update the version number to automatically regenerate the vocabulary.
    __version__: Optional[Any] = None

    def __init__(self, file_path: Union[Path, Mapping[str, Union[Path, List[Path]]], List[Path]],
                 name: Optional[str] = None, _vocab_path: Optional[Path] = None):
        """
        When subclassing :py:class:`DataLoader`, you should define your vocabularies in the subclass ``__init__``
        method, and **THEN** call the super-class ``__init__`` method.

        The super-class ``__init__`` method saves/loads dictionaries and calls the preprocessing function.

        :param file_path: Either a Path, or a dict mapping tags to paths, e.g. ``{'train': "...", 'test': "...", ...}``.
                          If this param is a dict, then the following methods should accept ``tag`` as a parameter:
                          - ``read_sentences``
                          - ``iterdata``
                          - ``iterbatch``
        """
        assert isinstance(file_path, (Path, dict, list))
        self.loader_name = name
        self.file_path = file_path
        if isinstance(self.file_path, dict):
            _file_paths = self.file_path.get('train', next(iter(self.file_path.values())))
        else:
            _file_paths = self.file_path
        if isinstance(_file_paths, Sequence):
            self._file_path: Path = _file_paths[0]
        else:
            self._file_path: Path = _file_paths

        self._specified_vocab_path = _vocab_path
        self._vocabs: Dict[str, Vocabulary] = {
            name: getattr(self, name)
            for name in dir(self) if isinstance(getattr(self, name), Vocabulary)
            # `type(x) is X` is incorrect because we have parameterized generic classes
        }
        loaded = self._load_vocabulary()
        if not loaded:
            self.generate_vocab()
            self._save_vocabulary()

        self.preprocess()

    def add_vocab(self, name: str, func: Optional[Callable[[], int]] = None) -> Vocabulary:
        vocab = Vocabulary[str, int](func)
        self._vocabs[name] = vocab
        return vocab

    def _vocab_path(self, name: str = '') -> Path:
        if self._specified_vocab_path is not None:
            path = self._specified_vocab_path
        elif self.loader_name is not None:
            path = path_add_suffix(self._file_path, self.loader_name + '.vocab')
        else:
            path = path_add_suffix(self._file_path, 'vocab')
        return path / name

    def _load_vocabulary(self) -> bool:
        if self.__version__ is not None:
            version_file = self._vocab_path('__version__')
            if not version_file.exists():
                return False
            with version_file.open('r') as f:
                version = f.read()
            if version != str(self.__version__):
                return False

        for name, vocab in self._vocabs.items():
            if not vocab._load or not os.path.exists(self._vocab_path(name)):
                continue
            with self._vocab_path(name).open('rb') as f:
                vocab.update(pickle.load(f), )
            # noinspection PyProtectedMember
            vocab.freeze()  # Make sure no modifications are made to the loaded dictionaries
            vocab.default_factory = None
            vocab._loaded = True
        exists = [self._vocab_path(name).exists() for name, vocab in self._vocabs.items() if vocab._load]
        return all(exists)

    def _save_vocabulary(self):
        self._vocab_path().mkdir(parents=True, exist_ok=True)
        if self.__version__ is not None:
            with self._vocab_path('__version__').open('w') as f:
                f.write(str(self.__version__))

        for name, vocab in self._vocabs.items():
            if not vocab._load or vocab._loaded:
                continue
            with self._vocab_path(name).open('wb') as f:
                pickle.dump(dict(vocab), f)

    def __len__(self) -> int:
        """
        Return dataset size. This is not really important and is only useful when you want a accurate epoch progress.

        However, it is still recommended to implement this as a sanity check.

        :return: Number of examples of the dataset.
        """
        raise NotImplementedError

    def read_example(self, tag: Optional[str] = None, verbose=False) -> Iterable[Any]:
        """
        Iterate over examples. You should read the files according to ``self.file_path`` (or ``self.file_path[tag]``
        if ``tag`` is specified).

        An ideal implementation should ensure that, for different datasets on the same task, this is the only method
        that requires rewriting.
        """
        raise NotImplementedError

    def generate_vocab(self):
        """
        Called when loading dataset for the first time. You should generate vocabularies and add special tokens
        in this function.

        Recommended implementation is by iterating through examples with ``read_sentence``, and store tokens into
        :class:`Vocabulary` by simply accessing it like a dictionary.

        Here's a non-comprehensive what you'll (probably) need to take care of:

        - character vocabularies
        - EOS, SOS, UNK tokens
        - word frequencies
        - other global attributes that are useful

        However, you should not store additional information in ``self``, because this method is not called if saved
        vocabularies are found. Instead, leave them to the ``preprocess`` function.
        """
        pass

    def preprocess(self):
        """
        Called after vocabulary is loaded or generated. This is when you read and preprocess your data.

        - For smaller datasets, you should read in all data.
        - For larger datasets, you should read on demand, and postpone preprocessing until ``iterdata`` is called.

        Regardless, here's what you would usually do:

        - generate reverse mappings of vocabularies (``id2word``)
        - store EOS, SOS tokens in ``self``

        It is also recommended that you add::

            # noinspection PyAttributeOutsideInit

        before the method definition to suppress inspection of assigning attributes to ``self`` outside ``__init__``.
        """
        pass

    def iterdata(self, tag: Optional[str] = None, *args, **kwargs) -> Iterable[List[Token]]:
        """
        Return an iterator allowing per-example iteration of data.

        For smaller datasets, you could simply return the list containing the whole dataset. This has benefits in
        e.g. performing full shuffle of data.
        """
        raise NotImplementedError

    @classmethod
    def _iterbatch_list(cls, data: List[List[Token]], size: int, shuffle=False, different_size=True) \
            -> Iterator[List[List[Token]]]:
        """
        List slice shuffling. Local order is preserved (useful for data presorted by sentence length).
        """
        length = len(data)
        idxs = (np.random.permutation if shuffle else np.arange)(utils.ceil_div(length, size)) * size
        for idx in idxs:
            batch = list(data[idx:(idx + size)])
            if idx + size > length and not different_size:
                batch.extend(list(data[0:(idx + size - length)]))
            yield batch

    @classmethod
    def _iterbatch_slide(cls, data_iter: Iterable[List[Token]], size: int, window_scale: int = 10) \
            -> Iterator[List[List[Token]]]:
        """
        Sliding window-based shuffling.
        """
        window_size = size * window_scale
        window = []
        for data in data_iter:
            window.append(data)
            if len(window) == window_size:
                np.random.shuffle(window)
                yield window[(-size):]
                window = window[:(window_size - size)]
        np.random.shuffle(window)
        shortage = len(window) % size
        window.extend(window[:shortage])
        for i in range(len(window) // size):
            yield window[(i * size):((i + 1) * size)]

    @classmethod
    def _iterbatch_iter(cls, data_iter: Iterable[List[Token]], size: int, different_size=False) \
            -> Iterator[List[List[Token]]]:
        """
        Simple batching with no reordering.
        """
        first_batch = None
        current_batch = []
        for data in data_iter:
            current_batch.append(data)
            if len(current_batch) == size:
                first_batch = first_batch or current_batch
                yield current_batch
                current_batch = []
        if first_batch is None:
            first_batch = current_batch  # when there's only one batch
        if len(current_batch) > 0:
            if not different_size:
                current_batch.extend(first_batch[:(size - len(current_batch))])
            yield current_batch

    @classmethod
    @overload
    def _convert_batch_tuple(cls, batch: List[Tuple[List[Token], ...]]) -> Tuple[List[List[Token]], ...]:
        ...

    @classmethod
    @overload
    def _convert_batch_tuple(cls, batch: List[List[Token]]) -> List[List[Token]]:
        ...

    @classmethod
    def _convert_batch_tuple(cls, batch):
        if not isinstance(batch[0], tuple):
            return batch
        tuple_size = len(batch[0])
        fields = [[] for _ in range(tuple_size)]
        for example in batch:
            for x in range(tuple_size):
                fields[x].append(example[x])
        return tuple(fields)

    def iterbatch(self, batch_size: int, tag: Optional[str] = None, shuffle=True, ordering='none',
                  *args, **kwargs) -> Iterator[List[List[Token]]]:
        """
        Common batching interface for all datasets. Three batching strategies are implemented:

        - ``_iterbatch_list``: List slice shuffling. List is chunked into batches while global order is preserved,
          and only the chunk indices are shuffled. This is the preferred strategy if indexing (``__getitem__``) is
          implemented for the `iterdata` return value.

        - ``_iterbatch_bucket``: Put sentences in buckets according to their length, so each batch only contains
          sentences of similar length.

        - ``_iterbatch_slide``: Partial shuffling with buffer. Data examples are stored in a buffer ``window_scale``
          times the batch size, and shuffling is performed on the buffer. This strategy is enabled if ``iterdata`` does
          not support indexing but shuffling is required.

        - ``_iterbatch_iter``: Iterator-based batching without shuffle. This is the fallback strategy is no shuffling
          is required.

        You can override this method to post-process batches, e.g. transposing and padding.

        Ordering: ``none``, ``sort``, ``bucket``.

        Note that ``kwargs`` is passed to ``iterdata`` only. In order to customize batching parameters, call underlying
        implementations explicitly.
        """
        data_iter = self.iterdata(tag, *args, **kwargs)
        if isinstance(data_iter, list):
            if ordering == 'sort':
                if isinstance(data_iter[0], tuple):
                    data_iter = sorted(data_iter, key=lambda x: len(x[0]), reverse=True)
                else:
                    data_iter = sorted(data_iter, key=len, reverse=True)
            batch_iter = self._iterbatch_list(data_iter, batch_size, shuffle=shuffle)
        else:
            if ordering != 'none':
                Logging.warn(f"Ordering '{ordering}' not supported for large datasets. "
                             f"If this is a small dataset, return a list instead of an iterator in `.iterdata`")
            if shuffle:
                batch_iter = self._iterbatch_slide(data_iter, batch_size)
            else:
                batch_iter = self._iterbatch_iter(data_iter, batch_size)

        # noinspection PyTypeChecker
        return utils.MeasurableGenerator(map(self._convert_batch_tuple, batch_iter),
                                         utils.ceil_div(len(data_iter), batch_size))

    @classmethod
    def sort_vocabulary(cls, word_vocab: Vocabulary[K, V], word_freq: Vocabulary[int, int]) \
            -> Tuple[Vocabulary[K, V], Vocabulary[int, int]]:
        """
        Sort vocabulary according to frequency. After sorting, most frequent word has index ``0``, and least frequent
        word has index ``len(vocab) - 1``.

        This is useful for adapted softmax e.g. :py:class:`torch.nn.AdaptiveLogSoftmaxWithLoss`.

        :type word_vocab: Vocabulary
        :type word_freq: Vocabulary

        :rtype: (Vocabulary, Vocabulary)
        :return: A pair of vocabularies (new_word_vocab, new_word_freq).
        """
        words = [(word, word_freq[idx]) for word, idx in word_vocab.items()]
        words = sorted(words, key=lambda x: x[1], reverse=True)

        new_word_vocab = Vocabulary[K, V]()
        new_word_freq = Vocabulary[int, int](int)
        for word, freq in words:
            new_word_freq[new_word_vocab[word]] = freq

        if word_vocab._is_frozen:
            new_word_vocab.freeze()
        if word_freq._is_frozen:
            new_word_freq.freeze()

        return new_word_vocab, new_word_freq

    @classmethod
    def prune_vocabulary(cls, word_vocab: Vocabulary[str, V], word_freq: Vocabulary[int, int], unk_token: str,
                         max_size: Optional[int] = None, min_freq: Optional[int] = None, verbose=False,
                         meta_data: Vocabulary[str, int] = None, keep_words: Optional[List[str]] = None) \
            -> Tuple[Vocabulary[str, V], Vocabulary[int, int]]:
        """
        Prune vocabulary according to frequency.

        :param word_vocab: Vocabulary to prune.
        :param word_freq: Vocabulary or list containing frequencies, indexed by values in `word_vocab`.
        :param unk_token: The UNK token used to replace the low frequency words.
        :param max_size: Maximum vocabulary size, or None if not constrained. If specified, only the `max_size` most
                   frequent words are kept.
        :param min_freq: Minimum frequency threshold, or None if not constrained. If specified, words with frequency
                   lower than `min_freq` are pruned.
        :param verbose: Whether to print statistics.
        :param meta_data: Vocabulary to store meta data (number of OOV words, cut-off frequency), or None if not needed.
        :param keep_words: List of words to keep regardless of constraints.

        :return: A pair of vocabularies (new_word_vocab, new_word_freq).
        """
        if max_size is not None or min_freq is not None:
            new_word_vocab = Vocabulary[str, V]()
            new_word_freq = Vocabulary[int, int](int)
            unk = new_word_vocab[unk_token]
            new_word_freq[unk] = 0
            for word in (keep_words or []):
                new_word_freq[new_word_vocab[word]] = word_freq[word_vocab[word]]

            combined_list = sorted(((word, word_freq[idx]) for word, idx in word_vocab.items()),
                                   key=lambda t: t[1], reverse=True)
            cut_off_idx = len(combined_list)
            if max_size is not None:
                cut_off_idx = min(cut_off_idx, max_size)
            if min_freq is not None:
                try:
                    index = next(idx for idx in range(len(combined_list)) if combined_list[idx][1] < min_freq)
                    cut_off_idx = min(cut_off_idx, index)
                except StopIteration:
                    pass  # no word has frequency less than `min_freq`
            threshold = combined_list[cut_off_idx - 1][1]

            for word, freq in combined_list[:cut_off_idx]:
                new_word_freq[new_word_vocab[word]] = freq
            new_word_freq[unk] = sum(freq for _, freq in combined_list[cut_off_idx:])
            num_oov = len(combined_list) - cut_off_idx

            if word_vocab._is_frozen:
                new_word_vocab.freeze()
            if word_freq._is_frozen:
                new_word_freq.freeze()

            if meta_data is not None:
                meta_data['num_oov'] = num_oov
                meta_data['cut_off_freq'] = threshold
            if verbose:
                Logging(2).log(f"Words: {len(new_word_freq):d} (+ {num_oov:d} UNK-ed, cut off at freq = {threshold:d})")
            return new_word_vocab, new_word_freq
        else:
            if meta_data is not None:
                meta_data['num_oov'] = 0
            if verbose:
                Logging(2).log(f"Words: {len(word_vocab):d} (no OOVs)")
            return word_vocab, word_freq


def read_file(path: Path) -> List[List[str]]:
    with path.open('r') as f:
        sents = [line.split() for line in f]
    return sents


def read_bitext_files(src_path: Path, tgt_path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    src_sents = read_file(src_path)
    tgt_sents = read_file(tgt_path)
    return src_sents, tgt_sents
