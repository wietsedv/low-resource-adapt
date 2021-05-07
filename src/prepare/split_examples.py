import json
from pathlib import Path
import random

factor = 0.25
random.seed(879324879)


def main():
    dirpath = Path('data/examples')
    for src_path in dirpath.glob('*_test.json'):
        tgt_path = src_path.parent / src_path.name.replace(
            '_test.json', '_dev1.json')
        tgt2_path = src_path.parent / src_path.name.replace(
            '_test.json', '_dev2.json')

        if tgt_path.exists():
            continue

        with open(src_path) as f:
            examples = [json.loads(line) for line in f]

        random.shuffle(examples)
        n = int(factor * len(examples))

        print(tgt_path)
        with open(tgt_path, 'w') as f:
            for ex in examples[:n]:
                f.write(f'{json.dumps(ex)}\n')

        print(tgt2_path)
        with open(tgt2_path, 'w') as f:
            for ex in examples[n:]:
                f.write(f'{json.dumps(ex)}\n')


if __name__ == '__main__':
    main()
