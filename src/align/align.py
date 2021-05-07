from pathlib import Path
import logging
import os

from transformers import BertModel, BertTokenizer
import numpy as np
import scipy as sp

logging.getLogger("transformers").setLevel(logging.ERROR)


def load_embeddings(model_path, suffixes=False, min_length=2):
    model_path = str(model_path)
    emb = BertModel.from_pretrained(
        model_path).embeddings.word_embeddings.weight.data.numpy()
    tokenizer = BertTokenizer.from_pretrained(model_path)

    special_tokens = tokenizer.all_special_tokens
    voc = {
        i: t
        for t, i in tokenizer.get_vocab().items()
        if t not in special_tokens and (
            suffixes or not t.startswith('##')) and len(t) >= min_length
    }

    voc_idx = sorted(voc.keys())
    emb = emb[voc_idx]
    voc = [voc[i] for i in voc_idx]
    return emb, voc


def calc_distances(src, tgt):
    dist = sp.spatial.distance.cdist(src, tgt, metric='euclidean')
    return dist


def align_tokens(dist_matrix, src_vocab, tgt_vocab):
    alignment_idx = np.argmin(dist_matrix, axis=0)
    alignment = []

    for tgt_i, tgt_token in enumerate(tgt_vocab):
        src_i = alignment_idx[tgt_i]
        src_token = src_vocab[src_i]
        alignment.append((tgt_token, src_token))
    return alignment


def main():
    base_path = Path('data') / 'output'

    mlm_version = 'v2'
    mlm_name = 'bertje_mlm_groningen'
    mlm_ckpt = 500_000

    src_model_path = 'GroNLP/bert-base-dutch-cased'

    tgt_path = Path('mlm') / mlm_version / mlm_name / f'checkpoint-{mlm_ckpt}'
    tgt_model_path = base_path / tgt_path

    src_embs, src_vocab = load_embeddings(src_model_path)
    print('src:', src_embs.shape, len(src_vocab))

    tgt_embs, tgt_vocab = load_embeddings(tgt_model_path)
    print('tgt:', tgt_embs.shape, len(tgt_vocab))

    dist_path = Path('data') / 'cache' / tgt_path / 'distances.npy'
    if dist_path.exists():
        print(f' > loading {dist_path}')
        dist_matrix = np.load(dist_path)
    else:
        print(f' > writing {dist_path}')
        dist_matrix = calc_distances(src_embs, tgt_embs)
        os.makedirs(dist_path.parent, exist_ok=True)
        np.save(dist_path, dist_matrix)

    out_path = Path('data') / 'cache' / tgt_path / 'alignment.tsv'
    alignment = align_tokens(dist_matrix, src_vocab, tgt_vocab)
    with open(out_path, 'w') as f:
        for tgt_token, src_token in alignment:
            f.write(f'{tgt_token}\t{src_token}\n')
    print(f' > {out_path}')


if __name__ == '__main__':
    main()
