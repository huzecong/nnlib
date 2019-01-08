__all__ = ['bieso2bio', 'bio2bieso']


def bieso2bio(tags):
    new_tags = []
    pos = -1
    for i, tag in enumerate(tags):
        if tag[0] in ['S', 'B']:
            pos = i
        elif tag[0] == 'O':
            pos = -1
            new_tags.append('O')
        if tag[0] in ['S', 'E']:
            typ = tags[pos][1:]
            new_tags.append('B' + typ)
            new_tags.extend(['I' + typ] * (i - pos))
    return new_tags


def bio2bieso(tags):
    new_tags = []
    pos = -1
    for i, tag in enumerate(tags):
        if tag[0] == 'B':
            pos = i
        elif tag[0] == 'O':
            pos = -1
            new_tags.append('O')
        if ((i == len(tags) - 1) or (tags[i + 1][0] != 'I')) and (pos != -1):
            typ = tags[pos][1:]
            if pos == i:
                new_tags.append('S' + typ)
            else:
                new_tags.append('B' + typ)
                new_tags.extend(['I' + typ] * (i - pos - 1))
                new_tags.append('E' + typ)
    return new_tags
