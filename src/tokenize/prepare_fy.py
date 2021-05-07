language = 'frisian'

src_path = f'data/sources/raw-{language}/plaintext-clean-withmarkers.txt'
src2_path = f'data/sources/raw-{language}/plaintext-oscar-clean.txt'
tgt_path = f'data/sources/raw-{language}/plaintext.txt'

with open(tgt_path, 'w') as f2:
    with open(src_path) as f:
        for line in f:
            if line == '\n':
                f2.write('\n')
            else:
                f2.write(line.rstrip() + ' ')

    with open(src2_path) as f:
        for line in f:
            f2.write(line)
