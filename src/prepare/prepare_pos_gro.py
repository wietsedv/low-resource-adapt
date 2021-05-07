import os
import json
from pathlib import Path

import pandas as pd

from .utils import POS_TAGSET

df = pd.read_excel('data/sources/Klunderloa6.xlsx',
                   skiprows=10,
                   names=['tokens', 'lemmas', 'pos_tags', 'lines', 'note'])

df = df.loc[df['lines'] != 0]

examples = []
tagset = set()
for _, group in df.groupby('lines'):
    # print(group, '\n')
    example = {
        'tokens': group['tokens'].tolist(),
        'pos_tags': group['pos_tags'].tolist(),
        'lemmas': group['lemmas'].tolist()
    }
    examples.append(example)
    tagset.update(example['pos_tags'])

print(f'{len(examples):,} examples')
print(f'{len(tagset):,} tags')

if len(tagset - POS_TAGSET) > 0:
    print('EXTRA TAGS:', tagset - POS_TAGSET)
if len(POS_TAGSET - tagset) > 0:
    print('MISSING TAGS:', POS_TAGSET - tagset)

path = Path('data/examples/gro_klunderloa_test.json')
os.makedirs(path.parent, exist_ok=True)
with open(path, 'w') as f:
    for ex in examples:
        f.write(json.dumps(ex) + '\n')

print(f'\nwritten to {path}')
