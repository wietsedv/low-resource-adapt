# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import os
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import List, Optional
import json

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from .trainer import Trainer
from .utils import LABEL_SETS

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        })
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    freeze_layers: Optional[List[str]] = field(
        default=None, metadata={"help": "Which layer(s) to freeze"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="pos",
        metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The name of the dataset to use (via the datasets library)."
        })
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "The configuration name of the dataset to use (via the datasets library)."
        })
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data file (a csv or JSON file)."
        })
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "An optional input evaluation data file to evaluate on (a csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json"
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json"
                ], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


def load_args():
    v_path = Path(sys.argv[1])
    d_path = Path(sys.argv[2])

    v_name = v_path.name.split('.')[0]
    d_name = d_path.name.split('.')[0]

    with open(v_path) as f:
        args = json.load(f)
    with open(d_path) as f:
        args.update(json.load(f))

    while (v_path.parent / 'config.json').exists():
        v_path = v_path.parent
        print(v_path)
        args_ = args
        with open(v_path / 'config.json') as f:
            args = json.load(f)
        args.update(args_)

    args['output_dir'] = os.path.join(args['output_dir'], v_name, d_name)
    if os.path.exists(args['output_dir']) and args['overwrite_output_dir']:
        ckpt = 0

        for ckpt_path in Path(args['output_dir']).glob('checkpoint-*'):
            this_ckpt = int(ckpt_path.name.split('-')[-1])
            if this_ckpt > ckpt:
                ckpt = this_ckpt

        if ckpt == 0:
            print('output dir exists, but does not contain a checkpoint')
            exit(1)
        ckpt_dir = os.path.join(args['output_dir'], f'checkpoint-{ckpt}')
        print('WARNING: continuing training', ckpt_dir)
        args['model_name_or_path'] = ckpt_dir
    return args


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))

    args = load_args()
    model_args, data_args, training_args = parser.parse_dict(args)

    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name,
                                data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = (f"{data_args.task_name}_tags"
                         if f"{data_args.task_name}_tags" in column_names else
                         column_names[1])

    label_names = LABEL_SETS[data_args.task_name]
    id2label = dict(enumerate(label_names))
    label2id = {l: i for i, l in id2label.items()}

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name else model_args.model_name_or_path,
        id2label=id2label,
        label2id=label2id,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Freeze layers
    if model_args.freeze_layers is not None:
        for name, p in model.named_parameters():
            if name in model_args.freeze_layers:
                p.requires_grad = False
                continue
            for lay_name in model_args.freeze_layers:
                if name.startswith(f'{lay_name}.'):
                    p.requires_grad = False
                    break

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # exit(0)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement")

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label2id[label[word_idx]] if data_args.
                                     label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = datasets.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Metrics
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            id2label[p] for prediction, label in zip(predictions, labels)
            for (p, lab) in zip(prediction, label) if lab != -100
        ]
        true_labels = [
            id2label[lab] for prediction, label in zip(predictions, labels)
            for (_, lab) in zip(prediction, label) if lab != -100
        ]

        mip, mir, mif, _ = precision_recall_fscore_support(true_labels,
                                                           true_predictions,
                                                           labels=label_names,
                                                           average='micro')
        map, mar, maf, _ = precision_recall_fscore_support(true_labels,
                                                           true_predictions,
                                                           labels=label_names,
                                                           average='macro')
        p, r, f, _ = precision_recall_fscore_support(true_labels,
                                                     true_predictions,
                                                     labels=label_names,
                                                     average=None)

        res = {
            "accuracy_score": accuracy_score(true_labels, true_predictions),
            "precision_micro": mip,
            "recall_micro": mir,
            "f1_micro": mif,
            "precision_macro": map,
            "recall_macro": mar,
            "f1_macro": maf
        }
        for i, tag in enumerate(label_names):
            res.update({
                f'precision_{tag}': p[i],
                f'recall_{tag}': r[i],
                f'f1_{tag}': f[i]
            })
        return res

    # Initialize our Trainer
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"]
        if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"]
        if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        model_path = (model_args.model_name_or_path if
                      (model_args.model_name_or_path is not None
                       and os.path.isdir(model_args.model_name_or_path)) else
                      None)
        trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate()

        output_eval_file = os.path.join(
            training_args.output_dir,
            f"eval_results_{data_args.task_name}.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in results.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        test_dataset = tokenized_datasets["test"]
        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [[
            id2label[p] for (p, lab) in zip(prediction, label) if lab != -100
        ] for prediction, label in zip(predictions, labels)]

        output_test_results_file = os.path.join(training_args.output_dir,
                                                "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir,
                                                    "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()