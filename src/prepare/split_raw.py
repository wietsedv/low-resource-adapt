import sys
from pathlib import Path
import random
import os

random.seed(69453879)

assert len(sys.argv) > 1
language = sys.argv[1]

src_path = f'data/sources/raw-{language}/plaintext.txt'
tgt_dir = Path('data') / 'plaintext-subsets'
os.makedirs(tgt_dir, exist_ok=True)

print(src_path)

with open(src_path) as f:
    docs = f.readlines()

doc_sizes = [len(doc.encode('utf-8')) / 1024 / 1024 for doc in docs]
max_size = sum(doc_sizes)
print(f'Maximum size: {max_size:.0f}mb')

for tgt_size in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
    if tgt_size > max_size:
        break

    tgt_path = tgt_dir / f'{language}-{str(tgt_size).zfill(2)}.txt'

    print(f'\nTarget size: {tgt_size}mb')

    cur_size = 0
    cur_docs = []
    idx = list(range(len(docs)))
    random.shuffle(idx)
    for i in idx:
        if (cur_size + doc_sizes[i]) > tgt_size:
            continue
        cur_size += doc_sizes[i]
        cur_docs.append(docs[i])

    if cur_size < (tgt_size - 0.5):
        print(
            f'Failure: current size is {cur_size:.1f}mb ({len(cur_docs)} docs')
        exit(1)

    print(f'Success! Current size is {cur_size:.1f}mb')
    with open(tgt_path, 'w') as f:
        f.writelines(cur_docs)
    print(tgt_path)
