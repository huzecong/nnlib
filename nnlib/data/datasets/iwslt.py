import itertools
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.error import HTTPError

import lxml.html

from . import download
from .dataset import NMTDataset
from ...utils import PathType


def _parse_iwslt_dataset(directory: Path):
    # Clean the dataset. Thanks to torchtext for this snippet:
    # https://github.com/pytorch/text/blob/ea64e1d28c794ed6ffc0a5c66651c33e2f57f01f/torchtext/datasets/translation.py#L152
    for xml_filename in directory.glob('*.xml'):
        txt_filename = xml_filename.with_suffix('')
        if txt_filename.is_file():
            continue

        with txt_filename.open('w') as f:
            root = lxml.html.parse(str(xml_filename)).getroot()[0][0][0]
            for doc in root.findall('doc'):
                for element in doc.findall('seg'):
                    f.write(element.text.strip() + '\n')

    xml_tags = [
        '</', '<url', '<keywords', '<talkid', '<description', '<reviewer', '<translator', '<title',
        '<speaker', '<transcript', '<doc'
    ]
    for original_filename in directory.glob('train.tags*'):
        txt_filename = original_filename.parent / original_filename.name.replace('.tags', '')
        if txt_filename.is_file():
            continue

        with txt_filename.open('w') as txt_file, original_filename.open('r') as original_file:
            for line in original_file:
                if not any(tag in line for tag in xml_tags):
                    txt_file.write(line.strip() + '\n')


class IWSLT(NMTDataset):
    """
    Data from `evaluation campaign <https://wit3.fbk.eu/>`_ of the International Workshop on Spoken Language
    Translation (IWSLT).
    """

    @classmethod
    def get_language_pairs(cls, *, version: int = None, **kwargs) -> List[Tuple[str, str]]:
        assert version is not None, "Must specify IWSLT version"
        if version == 2012:
            langs = ['ar', 'de', 'nl', 'pl', 'pt-br', 'ro', 'ru', 'sk', 'sl', 'tr', 'zh']
            lang_pairs = [('en', 'fr')] + [(lang, 'en') for lang in langs]
        elif version == 2013:
            langs = ['ar', 'de', 'es', 'fa', 'fr', 'it', 'nl', 'pl', 'pt-br', 'ro', 'ru', 'sl', 'tr', 'zh']
            lang_pairs = [(lang, 'en') for lang in langs] + [('en', lang) for lang in langs]
        elif version == 2014:
            langs = ['ar', 'de', 'es', 'fa', 'fr', 'he', 'it', 'nl', 'pl', 'pt-br', 'ro', 'ru', 'sl', 'tr', 'zh']
            lang_pairs = [(lang, 'en') for lang in langs] + [('en', lang) for lang in langs]
        elif version == 2015:
            langs = ['cs', 'de', 'fr', 'th', 'vi', 'zh']
            lang_pairs = [(lang, 'en') for lang in langs] + [('en', lang) for lang in langs]
        elif version == 2016:
            langs = ['ar', 'cs', 'de', 'fr']
            lang_pairs = [(lang, 'en') for lang in langs] + [('en', lang) for lang in langs]
        elif version == 2017:
            langs = ['de', 'en', 'it', 'nl', 'ro']
            lang_pairs = list(itertools.permutations(langs, 2))  # type: ignore
        elif version == 2018:
            lang_pairs = [('eu', 'en')]
        else:
            raise ValueError(f"Unsupported version number {version}. "
                             f"Currently versions 2012 through 2018 are supported.")
        return sorted(lang_pairs)

    # noinspection PyMethodOverriding
    @classmethod
    def load(cls, version: int, source: str, target: str, *,
             split: str = 'train', ver_tag: Optional[str] = None, directory: PathType = 'data/') \
            -> Tuple[Path, Path]:
        """
        :param version: IWSLT version (int), supports 2012 ~ 2018.
        :param source: Source language, two letter notation.
        :param target: Target language, two letter notation.
        :param split: Data split to load, ``train``, ``dev``, or ``test``.
        :param ver_tag: Tag for dev or test set (e.g. ``TED.dev2012``).
        :param directory: Save directory (and load from directory if possible).
        :return: Paths to selected data splits.
        """
        assert split in ['train', 'dev', 'test']

        folder_name_str = f'{source}-{target}'
        tgz_file = f'{source}-{target}'
        directory = Path(directory) / f'iwslt-{version}'
        wtf_iwslt2017_test_hack = False
        dev_tag = test_tag = ver_tag
        if version == 2012:
            release = '2012-03'
            dev_tag = dev_tag or 'TALK.dev2010'
            test_tag = test_tag or 'TALK.test2010'
        elif version == 2013:
            release = '2013-01'
            test_tag = test_tag or 'TED.tst2010'
        elif version == 2014:
            release = '2014-01'
        elif version == 2015:
            release = '2015-01'
        elif version == 2016:
            dev_tag = dev_tag or 'TED.dev2010'
            release = '2016-01'
        elif version == 2017:
            if split == 'test' and (test_tag is None or test_tag == 'TED.tst2017.mltlng'):
                # Logging.warn("IWSLT17 TED.tst2017.mltlng is a test-only dataset, "
                #              "it does not contain parallel sentences and thus cannot be used for evaluation. "
                #              "The \"paired\" paths are actually not paired data, "
                #              "so you'll likely see a BLEU score of zero.")
                wtf_iwslt2017_test_hack = True
                release = '2017-01-mted-test'
                test_tag = 'TED.tst2017.mltlng'
            else:
                release = '2017-01-trnmted'
                tgz_file = 'DeEnItNlRo-DeEnItNlRo'
                folder_name_str = 'DeEnItNlRo-DeEnItNlRo'
                dev_tag = dev_tag or 'TED.dev2010'
                test_tag = test_tag or 'TED.tst2010'
        elif version == 2018:
            release = '2018-01'
            tgz_file = 'eu-en'
            folder_name_str = 'IWSLT18.LowResourceMT.train_dev/eu-en'
            dev_tag = dev_tag or 'TED.dev2018'
        else:
            raise ValueError(f"Unsupported version number {version}. "
                             f"Currently versions 2012 through 2018 are supported.")

        if (source, target) not in cls.get_language_pairs(version=version):
            raise ValueError(f"The specified language pair ({source}-{target}) probably does not exist. "
                             f"Check <https://wit3.fbk.eu/mt.php?release={release}> for details. "
                             f"Also, run `IWSLT.get_language_pairs(version)` to see available language pairs.")

        short_ver = str(version)[-2:]

        train_filename = f'train.{source}-{target}.{{lang}}'
        url = f'https://wit3.fbk.eu/archive/{release}/texts/{tgz_file.replace("-", "/")}/{tgz_file}.tgz'
        dev_test_filename = f'IWSLT{short_ver}.{{ver_tag}}.{source}-{target}.{{lang}}'
        folder_name = Path(folder_name_str)
        ver_tag = dev_tag if split == 'dev' else test_tag
        check_files = [folder_name / (f'train.tags.{source}-{target}.{lang}' if split == 'train' else
                                      dev_test_filename.format(ver_tag=ver_tag, lang=lang))
                       for lang in [source, target]]

        if wtf_iwslt2017_test_hack:
            # hack here. wtf is wrong with iwslt
            url1 = f'https://wit3.fbk.eu/archive/{release}/texts/{source}/{target}/{source}-{target}.tgz'
            url2 = f'https://wit3.fbk.eu/archive/{release}/texts/{target}/{source}/{target}-{source}.tgz'
            for url in [url1, url2]:
                download.download_file_maybe_extract(url, directory=str(directory), check_files=[])
            test_filename = '{source}-{target}/IWSLT17.TED.tst2017.mltlng.{source}-{target}.{source}'
            _parse_iwslt_dataset(directory / f'{source}-{target}')
            _parse_iwslt_dataset(directory / f'{target}-{source}')

            source_filepath = directory / test_filename.format(source=source, target=target)
            target_filepath = directory / test_filename.format(source=target, target=source)

            return source_filepath, target_filepath

        # Taken from TorchNLP: https://github.com/PetrochukM/PyTorch-NLP/blob/master/torchnlp/datasets/iwslt.py
        try:
            download.download_file_maybe_extract(url=url, directory=str(directory), check_files=check_files)
        except (HTTPError, ValueError) as e:
            msg = f"HTTP error (code {e.code}) occurred" if isinstance(e, HTTPError) else f"Download check failed ({e})"
            # suppress previous exception
            raise ValueError(f"{msg}. The specified language pair ({source}-{target}) probably does not exist. "
                             f"Check <https://wit3.fbk.eu/mt.php?release={release}> for details.") from None

        directory = directory / folder_name

        _parse_iwslt_dataset(directory)

        if split == 'train':
            filename = train_filename
        else:
            valid_tags = list(set('.'.join(path.name.split('.')[1:3]) for path in directory.glob('IWSLT*')))
            if ver_tag is None:
                raise ValueError(f"IWSLT {version} does not contain a default {split} set tag, you must explicitly "
                                 f"specify `ver_tag`.\nAvailable choices are: {valid_tags}")
            if not (directory / dev_test_filename.format(ver_tag=ver_tag, lang=source)).exists():
                raise ValueError(f"IWSLT {version} does not contain the tag `{ver_tag}` that you specified.\n"
                                 f"Available choices are: {valid_tags}")
            filename = dev_test_filename.format(ver_tag=ver_tag, lang="{lang}")

        source_filepath = directory / filename.format(lang=source)
        target_filepath = directory / filename.format(lang=target)

        return source_filepath, target_filepath
