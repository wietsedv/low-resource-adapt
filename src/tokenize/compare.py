import pandas as pd
from transformers import BertTokenizer
from scipy.stats import pearsonr

src_tokenizers = [
    ('English', 'bert-base-cased'),
    ('German', 'bert-base-german-dbmdz-cased'),
    ('Dutch', 'GroNLP/bert-base-dutch-cased'),
    # ('Multi', 'bert-base-multilingual-cased'),
]

tgt_tokenizers = [('Gronings', 'data/tokenizers/bert-groningen-cased-10k'),
                  ('Frisian', 'data/tokenizers/bert-frisian-cased-10k')]

# df = pd.DataFrame({
#     'target': [lang for lang, _ in tgt_tokenizers for _ in src_tokenizers],
#     'source': [lang for _ in tgt_tokenizers for lang, _ in src_tokenizers],
#     'overlap':
#     None,
# })

# print(df)

src_accuracies = {
    'Dutch': 96.0,  # Alpino
    'English': 94.0,  # ParTUT
    'German': 94.0,  # HDT
}

orig_accuracies = {
    ('Gronings', 'Dutch'): 66.7,
    ('Gronings', 'English'): 33.5,
    ('Gronings', 'German'): 19.5,
    ('Frisian', 'Dutch'): 50.0,
    ('Frisian', 'English'): 37.1,
    ('Frisian', 'German'): 16.9,
}

accuracies = {
    ('Gronings', 'Dutch'): 92.4,
    ('Gronings', 'English'): 67.7,
    ('Gronings', 'German'): 86.7,
    ('Frisian', 'Dutch'): 95.4,
    ('Frisian', 'English'): 77.4,
    ('Frisian', 'German'): 89.0,
}

distances = {
    ('Gronings', 'Dutch'): 0.3873,
    ('Gronings', 'English'): 0.7263,
    ('Gronings', 'German'): 0.5381,
    ('Frisian', 'Dutch'): 0.5259,
    ('Frisian', 'English'): 0.7125,
    ('Frisian', 'German'): 0.5738,
}

rows = []
for src_lang, src_tokenizer in src_tokenizers:
    src_tokenizer = BertTokenizer.from_pretrained(src_tokenizer)
    src_vocab = src_tokenizer.get_vocab().keys()

    for tgt_lang, tgt_tokenizer in tgt_tokenizers:
        tgt_tokenizer = BertTokenizer.from_pretrained(tgt_tokenizer)
        tgt_vocab = tgt_tokenizer.get_vocab().keys()

        n = (len(tgt_vocab) - len(tgt_vocab - src_vocab)) / len(tgt_vocab)
        src_acc = src_accuracies[src_lang]
        orig_acc = orig_accuracies[(tgt_lang, src_lang)]
        acc = accuracies[(tgt_lang, src_lang)]

        rel_acc = acc / src_acc
        imp_acc = acc / orig_acc
        rows.append((tgt_lang, src_lang, n, src_acc, orig_acc, acc, rel_acc,
                     imp_acc, distances[(tgt_lang, src_lang)]))

df = pd.DataFrame(sorted(rows),
                  columns=[
                      'target', 'source', 'overlap', 'src_acc', 'orig_acc',
                      'acc', 'rel_acc', 'imp_acc', 'distance'
                  ])
print(df, end='\n\n')
print(df.corr(), end='\n\n')

dist = df['distance']
acc = df['acc']
print('Dist/acc:', pearsonr(df['distance'], df['acc']))
print('Overlap/acc:', pearsonr(df['overlap'], df['acc']))
