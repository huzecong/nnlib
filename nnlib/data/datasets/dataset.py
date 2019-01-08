import itertools
from pathlib import Path
from typing import List, Tuple, Union

from ...utils import PathType


class NMTDataset:
    @classmethod
    def get_languages(cls, *, _recursive=False, **kwargs) -> List[str]:
        if _recursive:
            raise NotImplementedError
        pairs = cls.get_language_pairs(_recursive=True, **kwargs)
        langs = [x for xs in pairs for x in xs]
        return sorted(set(langs))

    @classmethod
    def get_language_pairs(cls, *, _recursive=False, **kwargs) -> List[Tuple[str, str]]:
        if _recursive:
            raise NotImplementedError
        langs = cls.get_languages(_recursive=True, **kwargs)
        lang_pairs = list(itertools.permutations(langs, 2))
        # noinspection PyTypeChecker
        return lang_pairs  # type: ignore

    @classmethod
    def load(cls, *, split: str = 'train', directory: PathType = 'data/', **kwargs) \
            -> Union[Path, Tuple[Path, ...]]:
        """
        Return the path to the specific data split.

        :param split: Data split to load, usually ``train``, ``dev``, or ``test``.
        :param directory: Directory to store cached dataset.
        """
        raise NotImplementedError
