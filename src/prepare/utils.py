POS_TAGSET = {
    'CCONJ', 'SYM', 'ADV', 'NUM', 'PUNCT', 'AUX', 'VERB', 'PROPN', 'INTJ',
    'ADP', 'NOUN', 'SCONJ', 'ADJ', 'PRON', 'X', 'DET', 'PART'
}


def iter_conllu(path):
    tokens, pos_tags, lemmas = [], [], []
    with open(path) as f:
        for line in f:
            if line[0] in '\n#':
                if len(tokens) > 0:
                    yield {
                        'tokens': tokens,
                        'pos_tags': pos_tags,
                        'lemmas': lemmas
                    }
                    tokens, pos_tags, lemmas = [], [], []
                continue
            cols = line.split('\t')
            tokens.append(cols[1])
            lemmas.append(cols[2])
            if cols[3] == '_':
                cols[3] = 'X'
            pos_tags.append(cols[3])

    if len(tokens) > 0:
        yield {'tokens': tokens, 'pos_tags': pos_tags, 'lemmas': lemmas}
