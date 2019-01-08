from pathlib import Path
from typing import List, Optional

from .. import utils
from ..utils import PathType, path_add_suffix, path_lca

__all__ = ['tokenize', 'moses_tokenize', 'spacy_tokenize', 'spm_tokenize', 'get_tokenization_args']


def moses_tokenize(sents: List[str], lang: str) -> List[List[str]]:
    unsupported_langs = ['zh', 'ja', 'th']
    if lang.split('-')[0] in unsupported_langs:
        utils.Logging.warn(f"Moses does not support \"{lang}\" because it is not space-delimited. "
                           f"It will only split according to punctuation.")
    import sacremoses
    tok = sacremoses.MosesTokenizer(lang=lang)
    tok_sents = [tok.tokenize(sent.strip(), escape=False) for sent in sents]
    return tok_sents


def spacy_tokenize(sents: List[str], lang: str) -> List[List[str]]:
    import spacy
    try:
        nlp = spacy.load(lang.split('-')[0])
    except OSError:
        try:
            cls = spacy.util.get_lang_class(lang.split('-')[0])
            nlp = cls()
        except ImportError:
            utils.Logging.warn(f"spaCy does not support language \"{lang}\", falling back to default model")
            from spacy.lang.xx import MultiLanguage
            nlp = MultiLanguage()
        # tokenizing may require additional dependencies
    nlp('a')  # just run it first time otherwise it sometimes crashes for no reason
    tok_sents = []
    for sent in sents:
        sent = sent.strip()
        tokens = [token.text for token in nlp.make_doc(sent)] if sent != '' else []
        tok_sents.append(tokens)
    return tok_sents


def spm_tokenize(sents: List[str], spm_model: str) -> List[List[str]]:
    import sentencepiece
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(spm_model)
    tok_sents = []
    for sent in sents:
        sent = sent.strip()
        tokens = spm.EncodeAsPieces(sent) if sent != '' else []
        tok_sents.append(tokens)
    return tok_sents


def get_tokenization_args(tokenizer: Optional[str], lowercase: bool) -> str:
    return (tokenizer or '') + ('_lower' if lowercase else '')


def tokenize(path: Path, tokenizer: Optional[str], lang: Optional[str] = None, lowercase: bool = False,
             directory: PathType = 'data/tokenize', **tok_kwargs) -> Path:
    """
    Tokenize an input file or load the cached version.

    Available tokenizers and required fields:
    - `moses`: Moses from `sacremoses` package. Requires `lang`.
    - `spacy`: spaCy. Requires `lang`.
    - `spm`. SentencePiece. Requires `spm_model`: path to trained SPM model.

    :param path: Path to the input file.
    :param tokenizer: The tokenizer to use.
    :param lang: Language of the input file. Required for some tokenizers.
    :param lowercase: If true, lowercase each tokenized word.
    :param directory: Directory to store all the tokenized files.
    :param tok_kwargs: Additional arguments to pass to the tokenizer.
    :return: Path to the tokenized input file.
    """
    # shortcut to no tokenization
    if tokenizer is None and not lowercase:
        return path

    # check for invalid arguments
    valid_tokenizers = ['moses', 'spacy', 'spm']
    if tokenizer is not None and tokenizer not in valid_tokenizers:
        raise ValueError(f"Invalid tokenizer setting \"{tokenizer}\"")
    if tokenizer in ['moses', 'spacy'] and lang is None:
        raise ValueError(f"Must specify `lang` when using \"{tokenizer}\" tokenizer")
    if tokenizer == 'spm' and tok_kwargs.get('spm_model', None) is None:
        raise ValueError("Must supply `spm_model` as additional argument when using the SentencePiece tokenizer")

    # return cached file if exists
    base_path = Path(directory) / path_lca(path, directory)
    suffix = get_tokenization_args(tokenizer, lowercase)
    cached_path = path_add_suffix(base_path, suffix)
    if cached_path.exists():
        return cached_path

    # tokenize or re-use partially processed file
    tok_path = path_add_suffix(base_path, tokenizer) if tokenizer is not None else path
    if not tok_path.exists():
        with path.open('r') as f:
            sents = [line for line in f]
        with utils.work_in_progress(f"{tokenizer}: tokenizing {path}"):
            if tokenizer == 'moses':
                assert lang is not None
                tok_sents = moses_tokenize(sents, lang)
            elif tokenizer == 'spacy':
                assert lang is not None
                tok_sents = spacy_tokenize(sents, lang)
            elif tokenizer == 'spm':
                tok_sents = spm_tokenize(sents, tok_kwargs['spm_model'])
            else:
                assert False  # make the IDE happy
    else:
        with tok_path.open('r') as f:
            tok_sents = [line for line in f]

    # lowercase
    if lowercase:
        tok_sents = [[word.lower() for word in sent] for sent in tok_sents]

    # cache the file
    cached_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    with cached_path.open('w') as f:
        f.write('\n'.join(' '.join(sent) for sent in tok_sents))

    return cached_path
