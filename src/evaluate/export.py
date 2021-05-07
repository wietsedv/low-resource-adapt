from argparse import ArgumentParser
from pathlib import Path
import logging
import json

import yaml
from transformers import BertTokenizer, BertModel, BertForTokenClassification, BertTokenizerFast, TFBertForTokenClassification

logging.getLogger("transformers").setLevel(logging.ERROR)


def load_model(pos_version, pos_name, pos_ckpt, mlm_version, mlm_name,
               mlm_ckpt):

    model_path = None
    emb_path = None
    tokenizer_path = None

    for base_dir in ['data', 'data-per']:
        base_path = Path(base_dir) / 'output'

        model_path_ = base_path / 'pos' / pos_version / pos_name / f'checkpoint-{pos_ckpt}'
        tokenizer_path_ = model_path_

        if mlm_version is not None:
            emb_path_ = base_path / 'mlm' / mlm_version / mlm_name / f'checkpoint-{mlm_ckpt}'
            tokenizer_path_ = emb_path_

            if emb_path_.exists():
                emb_path = emb_path_

        if model_path_.exists():
            model_path = model_path_
        if tokenizer_path_.exists():
            tokenizer_path = tokenizer_path_

    if model_path is None:
        print(f'SKIPPING: pos model {model_path} does not exist')
        return None
    if mlm_version is not None and emb_path is None:
        print(f'SKIPPING: mlm model {emb_path} does not exist')
        return None
    if tokenizer_path is None:
        print(f'SKIPPING: tokenizer {tokenizer_path} does not exist')
        return None

    tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_path))
    model = BertForTokenClassification.from_pretrained(str(model_path))
    if mlm_version is not None:
        emb_model = BertModel.from_pretrained(str(emb_path))
        model.bert.embeddings = emb_model.embeddings

    model.config.vocab_size = len(tokenizer)

    return model, tokenizer


def export_model(pos_version, pos_name, pos_ckpt, mlm_version, mlm_name,
                 mlm_ckpt, lang):
    pos_base = pos_name.split('_')[0]
    mlm_base = mlm_name.split('_')[0]
    if mlm_version is not None and pos_base != mlm_base:
        return

    out_path = Path('data') / 'models' / (pos_name if lang is None else
                                          f'{pos_name}_{lang}')
    print(out_path)
    if out_path.exists():
        print(out_path, 'already exists. skipping')
        return

    model, tokenizer = load_model(pos_version, pos_name, pos_ckpt, mlm_version,
                                  mlm_name, mlm_ckpt)

    out_path = str(out_path)

    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    TFBertForTokenClassification.from_pretrained(
        out_path, from_pt=True).save_pretrained(out_path)
    BertTokenizer.from_pretrained(out_path).save_pretrained(out_path)

    out_path = Path(out_path)

    with open(out_path / 'config.json') as f:
        cfg = json.load(f)
    del cfg['_name_or_path']
    with open(out_path / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    with open(out_path / 'tokenizer_config.json') as f:
        cfg = json.load(f)
    del cfg['special_tokens_map_file']
    del cfg['name_or_path']
    with open(out_path / 'tokenizer_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)


def main():
    parser = ArgumentParser()
    parser.add_argument('-l', '--lang', required=True)
    parser.add_argument('-c', '--config', default='best')
    parser.add_argument('-m', '--model', default=['bertje_pos_nl_alpino'])
    args = parser.parse_args()

    cfg_path = f'config-{args.config}.yml' if args.config is not None else 'config.yml'
    with open(cfg_path) as f:
        cfg = yaml.full_load(f)

    for pos_version, pos_models in cfg['checkpoints']['pos'].items():
        for pos_name, pos_ckpts in pos_models.items():
            # if type(pos_ckpts) == int:
            #     pos_ckpts = [pos_ckpts]

            if pos_name not in args.model:
                continue

            pos_ckpts = [c for c in pos_ckpts if type(c) == dict]
            pos_ckpts = [c for c in pos_ckpts if args.lang in c]

            for pos_ckpt in pos_ckpts:
                if len(pos_ckpts) > 1 and ('best' not in pos_ckpt or args.lang
                                           not in pos_ckpt['best']):
                    continue

                export_model(
                    pos_version, pos_name,
                    pos_ckpt['pos'] if type(pos_ckpt) == dict else pos_ckpt,
                    None, 'none', 0, None)

                for mlm_version, mlm_models in cfg['checkpoints']['mlm'].items(
                ):
                    for mlm_name, mlm_ckpts in mlm_models.items():
                        if type(mlm_ckpts) == int:
                            mlm_ckpts = [mlm_ckpts]

                        mlm_lang = mlm_name.split('_')[-1]
                        if mlm_lang != args.lang:
                            continue

                        mlm_whitelist = None
                        # if type(pos_ckpt) != dict:
                        #     continue
                        if len(pos_ckpts) > 1 and ('best' not in pos_ckpt
                                                   or mlm_lang
                                                   not in pos_ckpt['best']):
                            continue

                        if type(pos_ckpt) == dict:
                            mlm_whitelist = pos_ckpt
                            ckpt = pos_ckpt['pos']
                        else:
                            ckpt = pos_ckpt

                        for mlm_ckpt in mlm_ckpts:
                            if mlm_version is not None and mlm_lang in mlm_whitelist and mlm_whitelist[
                                    mlm_lang] != mlm_ckpt:
                                continue
                            export_model(pos_version, pos_name, ckpt,
                                         mlm_version, mlm_name, mlm_ckpt,
                                         args.lang)


if __name__ == '__main__':
    main()
