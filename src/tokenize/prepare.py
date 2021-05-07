import sys

assert len(sys.argv) > 1
language = sys.argv[1]

src_path = f'data/sources/raw-{language}/plaintext-clean-withmarkers.txt'
tgt_path = f'data/sources/raw-{language}/plaintext.txt'

with open(src_path) as f, open(tgt_path, 'w') as f2:
    for line in f:
        if line == '\n':
            f2.write('\n')
        else:
            f2.write(line.rstrip() + ' ')
