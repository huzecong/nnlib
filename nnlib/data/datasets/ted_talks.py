import csv
import html
from pathlib import Path
from typing import List
from urllib.error import HTTPError

from nnlib.data.datasets import download
from nnlib.data.datasets.dataset import NMTDataset
from nnlib.utils.filesystem import PathType


def _parse_ted_talks_dataset(directory: Path, dataset_path: Path):
    with dataset_path.open('r') as f:
        reader = csv.reader(f, delimiter='\t')
        languages = next(reader)[1:]
        datasets: List[List[str]] = [[] for _ in languages]
        for row in reader:
            for sent, dataset in zip(row[1:], datasets):
                if sent == '__NULL__' or '_ _ NULL _ _' in sent:
                    sent = ''
                else:
                    sent = html.unescape(sent)
                dataset.append(sent)
    for lang, dataset in zip(languages, datasets):
        file_path = directory / lang
        with file_path.open('w') as f:
            f.write('\n'.join(dataset))


class TEDTalks(NMTDataset):
    r"""
    The TED talks multilingual dataset from:

    `[Qi et al. 2018] When and Why are Pre-trained Word Embeddings Useful for Neural Machine Translation?
    <https://arxiv.org/abs/1804.06323>`_

    .. note::
        This dataset is preprocessed by Moses. The Thai language (``th``) has also been tokenized.
    """

    @classmethod
    def get_languages(cls, **kwargs) -> List[str]:
        langs = ['ar', 'az', 'be', 'bg', 'bn', 'bs', 'cs', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi',
                 'fr', 'fr-ca', 'gl', 'he', 'hi', 'hr', 'hu', 'hy', 'id', 'it', 'ja', 'ka', 'kk', 'ko', 'ku', 'lt',
                 'mk', 'mn', 'mr', 'ms', 'my', 'nb', 'nl', 'pl', 'pt', 'pt-br', 'ro', 'ru', 'sk', 'sl', 'sq', 'sr',
                 'sv', 'ta', 'th', 'tr', 'uk', 'ur', 'vi', 'zh', 'zh-cn', 'zh-tw']
        return langs

    # noinspection PyMethodOverriding
    @classmethod
    def load(cls, language: str, split: str = 'train', directory: PathType = 'data/', **kwargs) -> Path:  # type: ignore
        r"""
        :param language: Language abbreviation, see `TED website
            <https://www.ted.com/participate/translate/our-languages>`_ for language names.
        :param split: Data split to load, ``train``, ``dev``, or ``test``.
        :param directory: Save directory (and load from directory if possible).
        :return: Paths to selected data splits.
        """
        assert split in ['train', 'dev', 'test']

        directory = Path(directory) / f'ted-talks-qi-2018'
        url = 'http://phontron.com/data/ted_talks.tar.gz'
        check_files = [f'all_talks_{tag}.tsv' for tag in ['train', 'dev', 'test']]

        try:
            download.download_file_maybe_extract(url=url, directory=str(directory), check_files=check_files)
        except (HTTPError, ValueError) as e:
            msg = f"HTTP error (code {e.code}) occurred" if isinstance(e, HTTPError) else f"Download check failed ({e})"
            # suppress previous exception
            raise ValueError(f"{msg}. Maybe the dataset was moved to another location. "
                             f"Check <https://github.com/neulab/word-embeddings-for-nmt> for details.") from None

        tag_folder = directory / split
        tag_folder.mkdir(parents=True, exist_ok=True)
        file_path = tag_folder / language
        if not file_path.exists():
            dataset_file = f'all_talks_{split}.tsv'
            _parse_ted_talks_dataset(tag_folder, directory / dataset_file)

        return file_path
