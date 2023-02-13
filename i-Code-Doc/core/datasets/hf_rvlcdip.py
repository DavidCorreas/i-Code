import os
import datasets
import requests
import torchvision.transforms as T
import torch
from collections import namedtuple
from torchvision.transforms import functional as F
from typing import Optional
from datasets import load_dataset  # type: ignore
from PIL.TiffImagePlugin import TiffImageFile
from io import BytesIO
import sys
sys.path.append('/workspaces/udop/i-Code-Doc')
from core.models import UdopDualForConditionalGeneration, UdopUnimodelForConditionalGeneration, UdopConfig, UdopTokenizer


def get_visual_bbox(image_size=224):
    image_feature_pool_shape = [image_size//16, image_size//16]
    visual_bbox_x = (torch.arange(
        0,
        1.0 * (image_feature_pool_shape[1] + 1),
        1.0,
    ) / image_feature_pool_shape[1])
    visual_bbox_y = (torch.arange(
        0,
        1.0 * (image_feature_pool_shape[0] + 1),
        1.0,
    ) / image_feature_pool_shape[0])
    visual_bbox_input = torch.stack(
        [
            visual_bbox_x[:-1].repeat(
                image_feature_pool_shape[0], 1),
            visual_bbox_y[:-1].repeat(
                image_feature_pool_shape[1], 1).transpose(
                    0, 1),
            visual_bbox_x[1:].repeat(
                image_feature_pool_shape[0], 1),
            visual_bbox_y[1:].repeat(
                image_feature_pool_shape[1], 1).transpose(
                    0, 1),
        ],
        dim=-1,
    ).view(-1, 4)
    return visual_bbox_input

def normalText(t):
    if type(t) is float:
        if t == int(t):
            t = int(t)
    t = str(t)
    return t.strip()

def request_ocr(url: str, image: TiffImageFile, lang='en', format='TIFF'):
    # Transform image to send in request
    byte_io = BytesIO()
    image.save(byte_io, format=format)
    byte_io.seek(0)
    
    response = requests.post(url, files={"image": byte_io}, data={"lang": "en"})
    return response.json()["result"]

def process_ocr(url, image: TiffImageFile, tokenizer, lang='en') -> Optional[tuple]:
    """
    Process TiffImageFile with OCR, tokenize the text and for every token create a bounding box.
    :param image: TiffImageFile. Image to process.
    :param tokenizer: Tokenizer. Tokenizer to use.
    :param image_size: int. Size of the image.
    :return: tuple. list_tokens, list_bboxes, image, page_size
    """
    text_list, bbox_list = [], []
    page_size = image.size

    try:
        result = request_ocr(url, image, lang)
    except Exception as e:
        print(f"OCR failed for {url}: {e}")
        return None
    
    # Normalize text and bbox
    for res in result:
        for line in res:
            text = line[1][0]
            n_text = normalText(text)
            if text == '':
                continue
            sub_tokens = tokenizer.tokenize(n_text)

            min_x = line[0][0][0]
            min_y = line[0][0][1]
            max_x = line[0][2][0]
            max_y = line[0][2][1]

            for sub_token in sub_tokens:
                text_list.append(sub_token)
                bbox_list.append([min_x, min_y, max_x, max_y])

    assert len(text_list) == len(bbox_list)

    return text_list, bbox_list, page_size


class Normalize(object):
    def __init__(self, mean, std, format='rgb'):
        self.mean = mean
        self.std = std
        self.format = format.lower()

    def __call__(self, image):
        if 'bgr' in self.format:
            image = image[[2, 1, 0]]
        if '255' in self.format:
            image = image * 255
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image

def img_trans_torchvision(image, image_size=224):
    trans = T.Compose([
            T.Resize([image_size,image_size]),
            T.ToTensor(),
            Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    image = trans(image)  # copy to make it writeable
    return image

def get_rvlcdip_labels():
    return [
        "letter",
        "form",
        "email",
        "handwritten",
        "advertisement",
        "scientific report",
        "scientific publication",
        "specification",
        "file folder",
        "news article",
        "budget",
        "invoice",
        "presentation",
        "questionnaire",
        "resume",
        "memo"
    ]

class HfRvlCdipDatasetBuilder:

    def __init__(self, data_args, tokenizer, num_proc=12):
        file_dir = "/workspaces/udop/i-Code-Doc/core/datasets"
        cache_dir = os.sep.join(file_dir.split(os.sep)[:-2]) + os.sep + '.hf_cache'
        
        self.ocr_url = "http://nginx:80/ocr"
        self.tokenizer = tokenizer
        self.max_seq_length = data_args.max_seq_length
        self.image_size = data_args.image_size
        self.num_proc = num_proc
        self.label_list = get_rvlcdip_labels()
        self.label_map = dict(zip(list(range(len(self.label_list))), self.label_list))
        
        # Load dataset
        self.dataset: datasets.DatasetDict = load_dataset("rvl_cdip", cache_dir=cache_dir) # type: ignore
        if data_args.max_samples > 0:
            self.dataset: datasets.DatasetDict = datasets.DatasetDict({key: self.dataset[key].select(range(data_args.max_samples)) for key in self.dataset.keys()})
        
        assert type(self.dataset) is datasets.dataset_dict.DatasetDict, f"Dataset is not of type Dataset, but {type(self.dataset)}"
        
        # Print size of dataset
        print(f"Train size: {len(self.dataset['train'])}")
        print(f"Validation size: {len(self.dataset['validation'])}")
        print(f"Test size: {len(self.dataset['test'])}")

    def build_dataset(self):
        print("Building dataset")
        dataset = self.dataset

        print("Processing OCR...")
        def read_image(example):
            result = process_ocr(self.ocr_url, example['image'], self.tokenizer)
            if result is None:
                return {
                "text_list": None,
                "bbox_list": None,
                "page_size": None
            }
            text_list, bbox_list, page_size = result
            return {
                "text_list": text_list,
                "bbox_list": bbox_list,
                "page_size": page_size
            }
        dataset = dataset.map(read_image, num_proc=self.num_proc)

        print("Filtering out None images and labels...")
        print(f'Number of examples before: {len(dataset["train"])}')
        filter_nones = lambda example: all(example[key] is not None for key in ['image', 'label', 'text_list', 'bbox_list', 'page_size'])
        dataset = dataset.filter(filter_nones)
        print(f'Number of examples after: {len(dataset["train"])}')
        # Example: {'image': TiffImageFile, 'label': int}
        print(dataset['train'][0].keys())
        # Example: {'image': TiffImageFile, 'label': int, 
        #   'text_list': list, 'bbox_list': list, 'page_size': (width, height)}
        print(dataset['train'][0].keys())
        
        print("Processing images...")
        process_image = lambda example: {
            "image": img_trans_torchvision(example['image'], self.image_size)
        }
        dataset = dataset.map(process_image, num_proc=self.num_proc)
        # Example: {'image': TiffImageFile, 'label': int,
        #   'text_list': list, 'bbox_list': list, 'page_size': (width, height)}
        print(dataset['train'][0].keys())

        print("Normalizing bounding boxes...")
        def normalize_bboxes(example):
            new_bboxes = []
            width, height = example['page_size']
            for bbox in example['bbox_list']:
                new_bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
                new_bboxes.append(new_bbox)
            return {
                "bbox_list": new_bboxes,
            }
        dataset = dataset.map(normalize_bboxes, num_proc=self.num_proc)
        # Example: {'image': TiffImageFile, 'label': int,
        #   'text_list': list, 'bbox_list': list, 'page_size': (width, height)}
        print(dataset['train'][0].keys())
        
        print("Adding visual bounding boxes...")
        add_visual_bboxes = lambda _: {"visual_seg_data": get_visual_bbox(self.image_size)}
        dataset = dataset.map(add_visual_bboxes, num_proc=self.num_proc)
        # Example: {'image': TiffImageFile, 'label': int,
        #   'text_list': list, 'bbox_list': list, 'page_size': (width, height),
        #   'visual_seg_data': torch.Tensor}
        print(dataset['train'][0].keys())
        
        print("Convert tokens to ids...")
        tokens_to_ids = lambda example: {
                "token_list": self.tokenizer.convert_tokens_to_ids(example['text_list']),
            }
        dataset = dataset.map(tokens_to_ids, num_proc=self.num_proc)
        # Example: {'image': TiffImageFile, 'label': int,
        #   'text_list': list, 'bbox_list': list, 'page_size': (width, height),
        #   'visual_seg_data': torch.Tensor, 'token_list': list}
        print(dataset['train'][0].keys())
        
        print("Convert to seq2seq...")
        def convert_seq_to_seq(example):
            prompt_text = 'document classification.'
            prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
            input_ids = prompt_ids + example['token_list']  # To add prompt to input ids
            bbox_list = [[0,0,0,0]] * len(prompt_ids) + example['bbox_list']  # To add prompt with empty bbox
            seq_labels = self.tokenizer(self.label_map[example["label"]], add_special_tokens=True) # To add EOS token
            
            attention_mask = [1] * len(input_ids)
            
            return {
                "input_ids": input_ids,
                "bbox_list": bbox_list,
                "seq_labels": seq_labels['input_ids'],
                "attention_mask": attention_mask,
                "decoder_attention_mask": seq_labels['attention_mask'],
            }
        dataset = dataset.map(convert_seq_to_seq, num_proc=self.num_proc)
        # Example: {'image': TiffImageFile, 'label': int,
        #   'text_list': list, 'bbox_list': list, 'page_size': (width, height),
        #   'visual_seg_data': torch.Tensor, 'token_list': list, 'prompt': int,
        #   'seq_labels': list, 'input_ids': list, 'attention_mask': list, 
        #   'decoder_attention_mask': list}
        print(dataset['train'][0].keys())

        add_char = lambda _: {"char_ids": [0]}
        dataset = dataset.map(add_char, num_proc=self.num_proc)
        add_char_seg_data = lambda _: {"char_seg_data": [[0,0,0,0]]}
        dataset = dataset.map(add_char_seg_data, num_proc=self.num_proc)
        # Example: {'image': TiffImageFile, 'label': int,
        #   'text_list': list, 'bbox_list': list, 'page_size': (width, height),
        #   'visual_seg_data': torch.Tensor, 'token_list': list, 'prompt': int,
        #   'seq_labels': list, 'input_ids': list, 'attention_mask': list, 
        #   'decoder_attention_mask': list, 'char_ids': torch.Tensor, 'char_seg_data': torch.Tensor}

        print("Shaping data...")
        def shape_data(example):
            return {
                "input_ids": example['input_ids'],
                "attention_mask": example['attention_mask'],
                "labels": example['seq_labels'],
                "seg_data": example['bbox_list'],
                "visual_seg_data": example['visual_seg_data'],
                "decoder_attention_mask": example['decoder_attention_mask'],
                "image": example['image'],
                "char_ids": example['char_ids'],
                "char_seg_data": example['char_seg_data']
            }
        dataset = dataset.map(shape_data, num_proc=self.num_proc, remove_columns=dataset["train"].column_names)
        dataset.set_format(type='torch')
        print(dataset['train'][0].keys())

        print("Final check...")
        print(f'Number of examples before: {len(dataset["train"])}')
        final_check = lambda example: all([
            len(example['input_ids']) == len(example['seg_data']),
            len(example['seg_data'].size()) == 2,
            len(example['char_seg_data'].size()) == 2
        ])
        dataset.filter(final_check)
        print(f'Number of examples after: {len(dataset["train"])}')

        return dataset

# Main
if __name__ == '__main__':

    tokenizer = UdopTokenizer.from_pretrained(
        "/workspaces/udop/i-Code-Doc/model/hf",
        cache_dir="/workspaces/udop/i-Code-Doc/.hf_cache/transformers",
        use_fast=True
    )

    DataArgs = namedtuple('DataArgs', ['max_samples', 'max_seq_length', 'image_size'])
    data_args = DataArgs(max_samples=-1, max_seq_length=512, image_size=224)

    new_dataset = HfRvlCdipDatasetBuilder(data_args, tokenizer, num_proc=20).build_dataset()
    new_dataset.save_to_disk("/workspaces/udop/i-Code-Doc/data/hf_rvl_cdip")
