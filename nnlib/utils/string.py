from typing import List

__all__ = ['ordinal', 'to_camelcase', 'to_underscore', 'to_capitalized']


def ordinal(x: int) -> str:
    suffix = 'th' if 10 <= x % 100 < 19 else {1: 'st', 2: 'nd', 3: 'rd'}.get(x % 10, 'th')
    return str(x) + suffix


def _get_name_pieces(name: str) -> List[str]:
    if '_' in name:
        # underscore
        return [x.lower() for x in name.split('_')]
    else:
        # camelcase / capitalized
        pos = [idx for idx, ch in enumerate(name) if ch.upper() == ch] + [len(name)]
        if pos[0] != 0:
            pos = [0] + pos
        return [name[l:r].lower() for l, r in zip(pos[:-1], pos[1:])]


def to_underscore(name: str) -> str:
    spl = _get_name_pieces(name)
    return '_'.join(spl)


def to_camelcase(name: str) -> str:
    spl = _get_name_pieces(name)
    if len(spl) == 0:
        return ''
    return spl[0] + ''.join(x.capitalize() for x in spl[1:])


def to_capitalized(name: str) -> str:
    spl = _get_name_pieces(name)
    return ''.join(x.capitalize() for x in spl)
