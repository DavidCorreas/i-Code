import numpy as np
import re
import PIL.ImageDraw
from transformers.integrations import WandbCallback
from transformers.utils import logging
from transformers import PreTrainedModel
from torch.utils.data import DataLoader
from datasets import Dataset
from config.hf_training_args import CustomTrainingArguments
from core.models.tokenization import UdopTokenizer
from torchvision import transforms


logger = logging.get_logger(__name__)


class UdopWandbCallback(WandbCallback):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("UdopWandbCallback initialized")
        self.table = self._wandb.Table(columns=["epoch", "prompt", "image", "label", "prediction"])

    def _create_rows(self, num_examples: int, epoch: str, dataloader: DataLoader, model: PreTrainedModel, tokenizer: UdopTokenizer, args: CustomTrainingArguments) -> list:
        # Select num_examples from dataloader randomly
        if args.only_grounding_samples:
            # Get a subset of the dataset to reduce the filtering time
            if num_examples > 100:
                examples = dataloader.dataset.shuffle().select(range(num_examples * 20))  # type: ignore
            else:
                examples = dataloader.dataset  # type: ignore
            # Filter out examples that don't have any grounding
            loc_inx_start = tokenizer.vocab_size - tokenizer._loc_extra_ids - tokenizer._other_extra_ids
            loc_inx_end = tokenizer.vocab_size - tokenizer._other_extra_ids
            # Check if any of the labels ids are between loc_inx_start and loc_inx_end
            def has_grounding(example):
                loc_inx = np.where(np.logical_and(example['labels'] >= loc_inx_start, example['labels'] < loc_inx_end))
                return len(loc_inx[0]) > 0
            dataset = examples.filter(has_grounding)
        else: 
            dataset = dataloader.dataset

        examples: Dataset = dataset.shuffle().select(range(num_examples))  # type: ignore

        def get_prediction(example):
            # Collate examples
            collate_example = dataloader.collate_fn([example])
            collate_example = {k: v.to(args.device) for k, v in collate_example.items()}  # type: ignore
            # Predict
            output = model(**collate_example).logits
            # Decode predictions
            prediction = tokenizer.batch_decode(output.argmax(-1), skip_special_tokens=False)[0]
            prediction = prediction.split(tokenizer.eos_token)[0] + tokenizer.eos_token
            del collate_example
            del output
            return {"wandb_prediction": prediction}
        examples = examples.map(get_prediction)

        # Create rows with epoch
        create_row = lambda e: {"wandb_epoch": epoch}
        examples = examples.map(create_row)

        # Add input_ids
        def get_prompt(example):
            # Get first [0,0,0,0] items of example['seg_data']
            first_ocr_idx = example['seg_data'].nonzero()[0][0]
            prompt = example['input_ids'][:first_ocr_idx]
            prompt = tokenizer.decode(prompt, skip_special_tokens=False)
            prompt = prompt.replace("Web action and object layout prediction. ", '')
            return {
                'wandb_prompt': prompt
            }
        examples = examples.map(get_prompt)

        # Add image with predictions
        def get_image(example):
            page_size = (1280, 720)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            inv_normalize = transforms.Normalize(
                mean=[-m/s for m, s in zip(mean, std)],
                std=[1/s for s in std]
            )
            norm_image = inv_normalize(example['image'])
            image = transforms.ToPILImage()(norm_image)
            res_img = image.resize(page_size)
            
            # Draw in the image the labels and predictions if exists
            # Count in example['labels'] the number of ocurrences of pattern <loc_
            draw = PIL.ImageDraw.Draw(res_img)
            label = tokenizer.decode(example['labels'], skip_special_tokens=False)
            num_loc = len(re.findall(r'<loc_', label))
            if num_loc == 4:
                bbox = tokenizer.convert_sentence_to_bbox(label, page_size)
                draw.rectangle(bbox, outline="red", width=5)
            # Draw prediction
            prediction = example['wandb_prediction'] if 'wandb_prediction' in example else ''
            num_loc = len(re.findall(r'<loc_', prediction))
            if num_loc == 4:
                bbox = tokenizer.convert_sentence_to_bbox(prediction, page_size)  # bbox = [x0, y0, x1, y1]
                # Check if y1 is greater than y0 and x1 is greater than x0
                if bbox[3] > bbox[1] and bbox[2] > bbox[0]:
                    draw.rectangle(bbox, outline="green", width=5)
                else:
                    # Draw message at the top
                    draw.text((0, 0), "Prediction is not valid", fill="green")
            
            del draw
            return {'wandb_image': res_img}  # Convert to wandb.Image later. Arrow can't serialize wandb.Image
        examples = examples.map(get_image)

        def get_label(example):
            label = tokenizer.batch_decode(example['labels'], skip_special_tokens=False)
            return {
                'wandb_label': label
            }
        examples = examples.map(get_label, batched=True)

        # Form rows
        examples.set_format('pandas')
        rows = []
        for i in range(len(examples)):
            rows.append([
                examples['wandb_epoch'][i],
                examples['wandb_prompt'][i],
                self._wandb.Image(examples['wandb_image'][i]),
                examples['wandb_label'][i],
                examples['wandb_prediction'][i]
        ])

        return rows

    def on_epoch_end(self, args: CustomTrainingArguments, state, control, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        if state.is_world_process_zero:
            epoch = f'{state.epoch}(train)' if state.epoch is not None else f'{-1}(train)'
            rows = self._create_rows(
                num_examples=args.samples_to_log_per_epoch,
                epoch=epoch,
                dataloader=kwargs["train_dataloader"],
                model=kwargs["model"],
                tokenizer=kwargs["tokenizer"],
                args=args
            )
            for row in rows:
                self.table.add_data(*row)

    def on_evaluate(self, args: CustomTrainingArguments, state, control, **kwargs):
        logger.info("UdopWandbCallback on_evaluate")
        super().on_evaluate(args, state, control, **kwargs)
        if state.is_world_process_zero:
            epoch = f'{state.epoch}(eval)' if state.epoch is not None else f'{-1}(eval)'
            rows = self._create_rows(
                num_examples=args.samples_to_log_per_eval,
                epoch=epoch,
                dataloader=kwargs["eval_dataloader"],
                model=kwargs["model"],
                tokenizer=kwargs["tokenizer"],
                args=args
            )
            for row in rows:
                self.table.add_data(*row)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        logger.info("UdopWandbCallback on_train_end")
        super().on_train_end(args, state, control, model=model, tokenizer=tokenizer, **kwargs)
        self._wandb.log({"Udop table": self.table})
