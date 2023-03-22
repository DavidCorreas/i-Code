from dataclasses import dataclass, field
from transformers.training_args import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):
    report_to: str = field(
        default="wandb",
        metadata={"help": "The service to report results to. Choices: wandb, comet, mlflow, tensorboard"},
    )
    examples_per_metrics: int = field(
        default=2,
        metadata={"help": "The number of examples to compute metrics on."},
    )
    samples_to_log_per_epoch: int = field(
        default=2,
        metadata={
            'help':
            'The number of samples to log per epoch for training examples, and per evaluation for evalutaion examples.'
            'If set to 0, no samples will be logged.'
        }
    )
    samples_to_log_per_eval: int = field(
        default=6,
        metadata={
            'help':
            'The number of samples to log per epoch for training examples, and per evaluation for evalutaion examples.'
            'If set to 0, no samples will be logged.'
        }
    )
    only_grounding_samples: bool = field(
        default=True,
        metadata={
            'help':
            'Whether to only log samples of object grounding.'
        }
    )