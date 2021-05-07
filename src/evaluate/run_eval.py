from argparse import ArgumentParser
from pathlib import Path
import json
import os
import logging

import yaml

from sklearn.metrics import classification_report
from tqdm import tqdm
import torch
from transformers import BertTokenizerFast, BertModel, BertForTokenClassification, TokenClassificationPipeline

logging.getLogger("transformers").setLevel(logging.ERROR)


def load_pipeline(pos_version, pos_name, pos_ckpt, mlm_version, mlm_name,
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

    model = BertForTokenClassification.from_pretrained(str(model_path))
    tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_path))
    if mlm_version is not None:
        emb_model = BertModel.from_pretrained(str(emb_path))
        model.bert.embeddings = emb_model.embeddings

    return TokenClassificationPipeline(model,
                                       tokenizer,
                                       task='pos',
                                       device=0,
                                       ignore_subwords=True)


def evaluate(clf, data_name, txt_path, json_path):
    y_true, y_pred = [], []
    with open(f'data/examples/{data_name}.json') as f:
        for line in tqdm(list(f), desc=data_name):
            ex = json.loads(line)
            y_true.extend(ex['pos_tags'])

            preds = [
                p for p in clf(' '.join(ex['tokens']))
                if not p['word'].startswith('##')
            ]
            pred_labels = [p['entity'] for p in preds]

            if len(pred_labels) > len(ex['pos_tags']):
                pred_labels = []
                for p in preds:
                    if len(ex['tokens']) == len(pred_labels):
                        break
                    if ex['tokens'][len(pred_labels)].startswith(p['word']):
                        pred_labels.append(p['entity'])

            if len(pred_labels) != len(ex['tokens']):
                print('invalid example:')
                # print(line)
                # print(pred_labels)
                print(ex['tokens'])
                print([p['word'] for p in preds])
                print(f'{len(pred_labels)} != {len(ex["tokens"])}')
                exit(1)

            y_pred.extend(pred_labels)

    report = classification_report(y_true, y_pred, zero_division=0)

    with open(json_path, 'w') as f:
        json.dump(classification_report(y_true,
                                        y_pred,
                                        output_dict=True,
                                        zero_division=0),
                  f,
                  indent=2)

    with open(txt_path, 'w') as f:
        f.write(report)
    print(report)


def run_eval(data_names, pos_version, pos_name, pos_ckpt, mlm_version,
             mlm_name, mlm_ckpt):
    pos_base = pos_name.split('_')[0]
    mlm_base = mlm_name.split('_')[0]
    if mlm_version is not None and pos_base != mlm_base:
        return

    out_dir = Path('data') / 'results'

    clf = None
    for data_name in data_names:
        out_path = out_dir / f'{pos_version}_{mlm_version}' / f'{pos_name}-{pos_ckpt}' / f'{mlm_name}-{mlm_ckpt}'
        os.makedirs(out_path, exist_ok=True)

        txt_path = out_path / f'{data_name}.txt'
        json_path = out_path / f'{data_name}.json'

        print(f'\n{txt_path}')

        if json_path.exists():
            print('already exists. skipping')
            continue

        if clf is None:
            clf = load_pipeline(pos_version, pos_name, pos_ckpt, mlm_version,
                                mlm_name, mlm_ckpt)
            if clf is None:
                print('could not load model')
                break

        with torch.no_grad():
            evaluate(clf, data_name, txt_path, json_path)


def main():
    parser = ArgumentParser()
    parser.add_argument('-l', '--lang', default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-c', '--config', default=None)
    args = parser.parse_args()

    cfg_path = f'config-{args.config}.yml' if args.config is not None else 'config.yml'
    with open(cfg_path) as f:
        cfg = yaml.full_load(f)

    data = cfg['data']['test' if args.test else 'dev']
    src_data = data
    if type(data) == dict:
        src_data = data['source'] + data['target']
        data = data['target']

    for pos_version, pos_models in cfg['checkpoints']['pos'].items():
        for pos_name, pos_ckpts in pos_models.items():
            if type(pos_ckpts) == int:
                pos_ckpts = [pos_ckpts]

            if args.test:
                pos_ckpts = [c for c in pos_ckpts if type(c) == dict]

                if args.lang is not None:
                    pos_ckpts = [c for c in pos_ckpts if args.lang in c]

            for pos_ckpt in pos_ckpts:
                if args.test and args.lang is not None:
                    if len(pos_ckpts) > 1 and ('best' not in pos_ckpt
                                               or args.lang
                                               not in pos_ckpt['best']):
                        continue

                if args.test:
                    run_eval(
                        src_data, pos_version, pos_name, pos_ckpt['pos'] if
                        type(pos_ckpt) == dict else pos_ckpt, None, 'none', 0)

                for mlm_version, mlm_models in cfg['checkpoints']['mlm'].items(
                ):
                    for mlm_name, mlm_ckpts in mlm_models.items():
                        if type(mlm_ckpts) == int:
                            mlm_ckpts = [mlm_ckpts]

                        mlm_lang = mlm_name.split('_')[-1]
                        if args.lang is not None and mlm_lang != args.lang:
                            continue

                        mlm_whitelist = None
                        if args.test:
                            if type(pos_ckpt) != dict:
                                continue
                            if len(pos_ckpts) > 1 and (
                                    'best' not in pos_ckpt
                                    or mlm_lang not in pos_ckpt['best']):
                                continue

                        if type(pos_ckpt) == dict:
                            mlm_whitelist = pos_ckpt
                            ckpt = pos_ckpt['pos']
                        else:
                            ckpt = pos_ckpt

                        for mlm_ckpt in mlm_ckpts:
                            if args.test and mlm_version is not None and mlm_lang in mlm_whitelist and mlm_whitelist[
                                    mlm_lang] != mlm_ckpt:
                                continue
                            run_eval(data, pos_version, pos_name, ckpt,
                                     mlm_version, mlm_name, mlm_ckpt)


if __name__ == '__main__':
    main()
