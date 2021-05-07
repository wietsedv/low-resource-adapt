from pathlib import Path
import os
import random

from conllu import parse

n_folds = 5
random.seed(9870534)

src_dir = Path('data/sources/ud-frisian')
tgt_dir = Path('data/conllu-folds/frisian')
os.makedirs(tgt_dir, exist_ok=True)

tokenlists = []
for filename in [
        'fy_testap.conllu', 'nieuws-combined.conllu', 'tresoar.conllu'
]:
    with open(src_dir / filename) as f:
        tokenlists_ = parse(f.read())
    print(filename, len(tokenlists_), tokenlists_[0], tokenlists_[0][0])
    tokenlists.extend(tokenlists_)

print('')
print(len(tokenlists), len(tokenlists) // n_folds, len(tokenlists) % n_folds)
random.shuffle(tokenlists)

fold_size = len(tokenlists) // n_folds

for i in range(1, n_folds + 1):
    if i == n_folds:
        fold = tokenlists[(i - 1) * fold_size:]
    else:
        fold = tokenlists[(i - 1) * fold_size:i * fold_size]

    print(i, len(fold))

    with open(tgt_dir / f'fold-{i}.conllu', 'w') as f:
        f.writelines([s.serialize() for s in fold])
