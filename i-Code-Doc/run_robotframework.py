#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import DatasetDict, load_from_disk
import evaluate
import nltk  # type: ignore
import wandb

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    EvalPrediction,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from core.trainers import DataCollator, UdopWandbCallback
from core.models import UdopDualForConditionalGeneration, UdopUnimodelForConditionalGeneration, UdopConfig, UdopTokenizer
from config.hf_training_args import CustomTrainingArguments


MODEL_CLASSES = {
    'UdopDual': (UdopConfig, UdopDualForConditionalGeneration, UdopTokenizer),
    'UdopUnimodel': (UdopConfig, UdopUnimodelForConditionalGeneration, UdopTokenizer),
}

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "local dataset stored location"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    image_size: Optional[int] = field(
    default=512,
    metadata={
        "help": "image size"
        "value if set."
    },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            'help':
            'The maximum total input sequence length after tokenization. Sequences longer '
            'than this will be truncated, sequences shorter will be padded.'
        },
    )   
    max_seq_length_decoder: int = field(
        default=16,
        metadata={
            'help':
            'The maximum total input sequence length after tokenization. Sequences longer '
            'than this will be truncated, sequences shorter will be padded.'
        },
    )  


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        # default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        # default=None, 
        metadata={'help': 'Model type selected in the list.'})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )    
    attention_type: str = field(
        default="original_full",
        metadata={"help": "Attention type: BigBird configuruation only. Choices: block_sparse (default) or original_full"},
    )


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.logging_dir = os.path.join(training_args.output_dir, 'runs')
    if model_args.cache_dir is None:
        model_args.cache_dir = os.path.join(training_args.output_dir, 'cache')
    os.makedirs(model_args.cache_dir, exist_ok=True)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Set weights and biases if available
    if "wandb" in training_args.report_to:
        # set the wandb project where this run will be logged
        os.environ["WANDB_PROJECT"]="udop"
        # save your trained model checkpoint to wandb
        os.environ["WANDB_LOG_MODEL"]="true"
        # turn off watch to log faster
        os.environ["WANDB_WATCH"]="false"

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Model arguments: {model_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

 
    #if 'local' in model_args.model_name_or_path:
    if model_args.model_type in MODEL_CLASSES:
        config_type, model_type, tokenizer_type = MODEL_CLASSES[model_args.model_type]
    else:
        config_type, model_type, tokenizer_type = AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = config_type.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        attention_type=model_args.attention_type if model_args.attention_type else None,
    )
    tokenizer = tokenizer_type.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Check if model_args.model_name_or_path is a path or url to a directory containing tokenizer files
    if os.path.isdir(model_args.model_name_or_path):
        config.mae_checkpoint = os.path.join(model_args.model_name_or_path, config.mae_checkpoint)
    model = model_type.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

   # Get datasets
    dataset = load_from_disk(data_args.dataset_name)
    assert isinstance(dataset, DatasetDict), "load_dataset should return a dict"
    train_dataset = dataset['train']
    if training_args.do_train:
        assert train_dataset is not None, "Training requires a train dataset"
    eval_dataset = dataset['validation']
    if training_args.do_eval:
        assert eval_dataset is not None, "Evaluation requires an evaluation dataset"
    test_dataset = dataset['test']
    if training_args.do_predict:
        assert test_dataset is not None, "Prediction requires a test dataset"               

    # Data collator
    padding = "max_length" if data_args.pad_to_max_length else False
    data_collator = DataCollator(
        tokenizer=tokenizer,
        padding=padding,
        max_length=data_args.max_seq_length,
        max_length_decoder=data_args.max_seq_length_decoder,
    )

    # metric = evaluate.load("accuracy")
    # Setup evaluation
    nltk.download("punkt", quiet=True)
    rouge_metric = evaluate.load("rouge")
    mean_iou = evaluate.load("mean_iou")
    assert rouge_metric is not None, "Could not load metric"
    # Setup executor to add rows to the wandb table
    if training_args.report_to == "wandb":
        global table_rows
        table_rows = []

    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # decode preds and labels
        assert tokenizer.pad_token_id is not None, "Please make sure that `tokenizer.pad_token_id` is defined."
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # iou metric
        if isinstance(tokenizer, UdopTokenizer):
            # Get bbox if 
            pred_bbox = tokenizer.get_bbox_from_logits(predictions)
            label_bbox = tokenizer.get_bbox_from_logits(labels)
            remove_idx = np.where(label_bbox[:, 0] == -1)[0]
            pred_bbox = np.delete(pred_bbox, remove_idx, axis=0)
            label_bbox = np.delete(label_bbox, remove_idx, axis=0)
            shape = (pred_bbox.shape[0], tokenizer._loc_extra_ids, tokenizer._loc_extra_ids)
            pred_map = np.zeros(shape)
            label_map = np.zeros(shape)
            for i in range(label_bbox.shape[0]):
                if pred_bbox[i, 0] == -1:
                    continue
                pred_map[i, pred_bbox[i, 1]:pred_bbox[i, 3], pred_bbox[i, 0]:pred_bbox[i, 2]] = 1
                label_map[i, label_bbox[i, 1]:label_bbox[i, 3], label_bbox[i, 0]:label_bbox[i, 2]] = 1
            iou_result = mean_iou.compute(predictions=pred_map, references=label_map, num_labels=1, ignore_index=-1)  # Takes long time
            if iou_result:
                # Remove metrics that are categories of classes. Not necessary in binary detection
                iou_result = {k: v for k, v in iou_result.items() if not isinstance(v, np.ndarray)}
                result = {**result, **iou_result} if result else iou_result

        return result

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,  # type: ignore[arg-type]
        eval_dataset=eval_dataset if training_args.do_eval else None,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # type: ignore[arg-type]
        callbacks=[UdopWandbCallback()] if "wandb" in training_args.report_to else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(test_dataset)  # type: ignore[arg-type]
        predictions = np.argmax(predictions, axis=-1)
        
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        
        # decode preds and labels
        assert tokenizer.pad_token_id is not None, "Please make sure that `tokenizer.pad_token_id` is defined."
        assert labels is not None, "Please make sure that `labels` is defined."
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Get eos_token id
        decoded_preds = tokenizer.batch_decode(predictions)
        decoded_preds = [pred.split(tokenizer.eos_token)[0] for pred in decoded_preds]
        decoded_labels = tokenizer.batch_decode(labels)
        decoded_labels = [label.split(tokenizer.eos_token)[0] for label in decoded_labels]

        true_predictions = [f"Pred: {p}, Label: {l}" for (p, l) in zip(decoded_preds, decoded_labels)]
        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(prediction + "\n")


if __name__ == "__main__":
    main()
