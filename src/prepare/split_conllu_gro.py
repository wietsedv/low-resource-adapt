from pathlib import Path
import os
import random

from conllu.models import Token, TokenList
import pandas as pd

n_folds = 5
random.seed(9870534)

tgt_dir = Path('data/conllu-folds/groningen')
os.makedirs(tgt_dir, exist_ok=True)

df = pd.read_excel('data/sources/Klunderloa6.xlsx',
                   names=['form', 'lemma', 'upos', 'sent_id', 'note'])

# df = df.loc[df['id'] != 0]

tokenlistss = []
prev_sent_id = 0
id_ = 0

for _, row in df.iterrows():
    if row['sent_id'] == 0:
        if str(row['form']).startswith('# titel:'):
            tokenlistss.append([])
        continue

    if row['sent_id'] != prev_sent_id:
        tokenlistss[-1].append([])
        prev_sent_id = row['sent_id']
        id_ = 0

    id_ += 1

    token = Token({
        'id': id_,
        'form': str(row['form']),
        'lemma': str(row['lemma']),
        'upos': str(row['upos']),
        'xpos': None,
        'feats': None,
        'head': None,
        'deprel': None,
        'deps': None,
        'misc': None
    })
    tokenlistss[-1][-1].append(token)

tokenlistss = [[TokenList(tokenlist) for tokenlist in tokenlists]
               for tokenlists in tokenlistss]

print(len(tokenlistss))
print([len(tlists) for tlists in tokenlistss])

print(tokenlistss[0][0], tokenlistss[0][0][0], tokenlistss[0][0][1])

# print('')
# print(len(tokenlists), len(tokenlists) % n_folds)
# random.shuffle(tokenlists)

fold_size = len(tokenlistss) // n_folds

for i in range(1, n_folds + 1):
    if i == n_folds:
        tlistss = tokenlistss[(i - 1) * fold_size:]
    else:
        tlistss = tokenlistss[(i - 1) * fold_size:i * fold_size]

    fold = [tlist for tlists in tlistss for tlist in tlists]

    print(i, len(fold))

    with open(tgt_dir / f'fold-{i}.conllu', 'w') as f:
        f.writelines([s.serialize() for s in fold])
