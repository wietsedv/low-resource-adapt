from pathlib import Path
import os
import sys

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, BertConfig

assert len(sys.argv) > 1
language = sys.argv[1]

src_path = f'data/sources/raw-{language}/plaintext-clean-withmarkers.txt'
out_dir = Path('data') / 'tokenizers-new'

for vocab_size in [10, 20, 30]:
    print(f' > {vocab_size}k')
    path = out_dir / f'bert-{language}-cased-{vocab_size}k' / 'tokenizer.json'

    if path.exists():
        print(f'{path} already exists. skipping')
        continue

    os.makedirs(path.parent, exist_ok=True)

    tokenizer = BertWordPieceTokenizer(lowercase=False, strip_accents=False)
    tokenizer.train(src_path, vocab_size=vocab_size * 1000, min_frequency=100)
    tokenizer.save(str(path), pretty=True)

    tok_dir = str(path.parent)
    BertTokenizerFast.from_pretrained(
        tok_dir, do_lower_case=False,
        strip_accents=False).save_pretrained(tok_dir)

    config = BertConfig.from_pretrained('bert-base-cased',
                                        vocab_size=vocab_size * 1000)
    config.save_pretrained(str(path.parent))

    print(tok_dir)
