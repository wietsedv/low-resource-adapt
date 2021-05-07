import os
import json
from pathlib import Path

from .utils import POS_TAGSET, iter_conllu

for treebank_name in ['Alpino', 'LassySmall']:
    src_dir = Path(
        'data/sources') / 'ud-treebanks-v2.7' / f'UD_Dutch-{treebank_name}'

    tgt_dir = Path('data/examples')
    os.makedirs(tgt_dir.parent, exist_ok=True)

    for fold in ['train', 'dev', 'test']:
        src_path = src_dir / f'nl_{treebank_name.lower()}-ud-{fold}.conllu'
        tgt_path = tgt_dir / f'nl_{treebank_name.lower()}_{fold}.json'

        examples = [ex for ex in iter_conllu(src_path)]
        tagset = {t for ex in examples for t in ex['pos_tags']}

        if len(tagset - POS_TAGSET) > 0:
            print('EXTRA TAGS:', tagset - POS_TAGSET)
        if len(POS_TAGSET - tagset) > 0:
            print('MISSING TAGS:', POS_TAGSET - tagset)

        with open(tgt_path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

        print(f'{len(examples):,} examples written to {tgt_path}')
