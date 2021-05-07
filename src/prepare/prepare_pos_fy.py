import os
import json
from pathlib import Path

from .utils import POS_TAGSET, iter_conllu

src_dir = Path('data/sources') / 'ud-frisian'

tgt_dir = Path('data/examples')
os.makedirs(tgt_dir.parent, exist_ok=True)

full_examples = []
for fold, name in [('fy_testap', 'testap'), ('nieuws-combined', 'nieuws'),
                   ('tresoar', 'tresoar')]:
    src_path = src_dir / f'{fold.lower()}.conllu'
    tgt_path = tgt_dir / f'fy_{name}_test.json'

    examples = [ex for ex in iter_conllu(src_path)]
    tagset = {t for ex in examples for t in ex['pos_tags']}

    full_examples.extend(examples)

    if len(tagset - POS_TAGSET) > 0:
        print('EXTRA TAGS:', tagset - POS_TAGSET)
    if len(POS_TAGSET - tagset) > 0:
        print('MISSING TAGS:', POS_TAGSET - tagset)

    with open(tgt_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f'{len(examples):,} examples written to {tgt_path}')

tgt_path = tgt_dir / 'fy_full_test.json'
with open(tgt_path, 'w') as f:
    for ex in full_examples:
        f.write(json.dumps(ex) + '\n')

print(f'{len(full_examples):,} examples written to {tgt_path}')
