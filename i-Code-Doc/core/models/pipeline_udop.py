import io
import torch
import torchvision.transforms as T
import requests
import re
from typing import List, Optional, Tuple
from PIL import Image
from torchvision.transforms import functional as F
from transformers import Pipeline
from transformers.image_utils import load_image


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

def request_ocr(url: str, image: Image.Image, lang='en', format='PNG'):
    # Transform image to send in request
    byte_io = io.BytesIO()
    image.save(byte_io, format=format)
    byte_io.seek(0)
    
    response = requests.post(url, files={"image": byte_io}, data={"lang": "en"})
    return response.json()["result"]

def process_ocr(url, image: Image.Image, tokenizer, lang='en') -> Optional[tuple]:
    """
    Process TiffImageFile with OCR, tokenize the text and for every token create a bounding box.
    :param image: Image. Image to process.
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

def img_trans_torchvision(image, image_size=224) -> torch.Tensor:
    trans = T.Compose([
            T.Resize([image_size,image_size]),
            T.ToTensor(),
            Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    image = trans(image)  # copy to make it writeable
    return image


class UdopPipeline(Pipeline):
    """
    Pipeline to do inference with Udop.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.page_size = None

    def _sanitize_parameters(self, 
        image_size: int = 224, 
        ocr_url: Optional[str] = 'http://nginx_udop:80/ocr',
        max_seq_len: Optional[int] = None,
        word_boxes: Optional[Tuple[List[str], List[Tuple[int, int, int, int]]]] = None,
        lang: Optional[str] = None,
        max_new_tokens: Optional[int] = 50,
        **kwargs):

        preprocess_kwargs = {}
        
        preprocess_kwargs["image_size"] = image_size
        # If not ocr_url provided, check if word_boxes are provided
        if ocr_url is None:
            if word_boxes is not None:
                preprocess_kwargs["word_boxes"] = word_boxes
            else:
                raise ValueError("Please provide either an ocr_url or word_boxes")
                
        else:
            preprocess_kwargs["ocr_url"] = ocr_url
            preprocess_kwargs["lang"] = lang
        
        assert self.tokenizer is not None, "Please provide a tokenizer when instantiating the pipeline"
        preprocess_kwargs["max_seq_len"] = max_seq_len if max_seq_len is not None else self.tokenizer.model_max_length

        process_kwargs = {}
        process_kwargs["max_new_tokens"] = max_new_tokens

        return preprocess_kwargs, process_kwargs, {}

    def preprocess(self, 
        inputs: dict,
        image_size: int = 224,
        ocr_url: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        word_boxes: Optional[Tuple[List[str], List[Tuple]]] = None,
        lang: Optional[str] = None
    ):
        assert self.tokenizer is not None, "Please provide a tokenizer when instantiating the pipeline"
        if max_seq_len is None:
            max_seq_len = self.tokenizer.model_max_length
        
        # Load image
        img: Image.Image = None  # type: ignore
        if inputs.get("image", None) is not None:
            img = load_image(inputs["image"])
        else:
            raise ValueError("No image provided")

        # Read image
        self.page_size = img.size
        text_list = []
        bbox_list = []
        if word_boxes is None:
            assert ocr_url is not None, "Please provide an Paddle OCR url when instantiating the pipeline"
            lang = "en" if lang is None else lang
            result = process_ocr(ocr_url, img, self.tokenizer, lang)
            assert result is not None, "OCR failed"
            text_list, bbox_list, _ = result
        else:
            text_list, bbox_list = word_boxes

        # Process image
        image = img_trans_torchvision(img, image_size=image_size)

        # Normalize bbox
        norm_bbox_list = []
        width, height = self.page_size
        for bbox in bbox_list:
            norm_bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            norm_bbox_list.append(norm_bbox)
        bbox_list = norm_bbox_list
        
        # Adding visual bbox
        visual_seg_data = get_visual_bbox(image_size)

        # Convert to token ids
        token_list = self.tokenizer.convert_tokens_to_ids(text_list)
        token_list = [token_list] if isinstance(token_list, int) else token_list

        # Making prompt
        prompt_text = "Web action and object layout prediction."
        instruction = inputs['instruction']
        prompt = prompt_text + " " + instruction
        # Encoder
        prompt_ids =  self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = prompt_ids + token_list
        input_ids = input_ids[:max_seq_len]
        bbox_list = [[0,0,0,0]] * len(prompt_ids) + bbox_list
        attention_mask = [1] * len(input_ids)
        char_ids = [0]
        char_seg_data = [[0,0,0,0]]

        # Model input
        model_input = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
            "seg_data": torch.tensor([bbox_list]),
            # Add one more dimension for batch size
            "image": image[None, :],
            "visual_seg_data": visual_seg_data[None, :],
            "char_ids": torch.tensor([char_ids]),
            "char_seg_data": torch.tensor([char_seg_data])
        }    
        
        return model_input

    def _forward(self, model_inputs, max_new_tokens=100):
        outputs = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        return outputs

    def postprocess(self, model_outputs):
        assert self.tokenizer is not None, "Please provide a tokenizer when instantiating the pipeline"
        decoded_preds = self.tokenizer.batch_decode(model_outputs)
        decoded_preds = [pred.split(self.tokenizer.eos_token)[0] for pred in decoded_preds]
        pred = decoded_preds[0]
        # Remove special tokens
        pred = re.sub(self.tokenizer.eos_token, '', pred)
        pred = re.sub(self.tokenizer.pad_token, '', pred)
        # Replace <loc> with bbox
        loc = re.findall(r'<loc_(\d+)>', pred)
        if len(loc) > 0:
            bbox = self.tokenizer.convert_sentence_to_bbox(pred, self.page_size)  # type: ignore
            pred = re.sub(r'<loc_\d+>', '', pred)
            pred += f" (x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]})"

        return pred
    

if __name__ == "__main__":
    import sys
    sys.path.append("/workspaces/udop/i-Code-Doc")

    import os
    import base64
    import PIL.Image
    from datasets import load_dataset, DatasetDict
    from core.datasets.robotframework import UdopExampleToInstruction
    from core.models.udop_unimodel import UdopUnimodelForConditionalGeneration
    from core.models.tokenization import UdopTokenizer

    # Load model
    model = UdopUnimodelForConditionalGeneration.from_pretrained("/workspaces/udop/i-Code-Doc/finetune_robotframework")
    tokenizer = UdopTokenizer.from_pretrained("/workspaces/udop/i-Code-Doc/finetune_robotframework")
    udop = UdopPipeline(model=model, tokenizer=tokenizer)
    # Load example of datset
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.sep.join(file_dir.split(os.sep)[:-2]) + os.sep + '.hf_cache'
    dataset = load_dataset("json", data_dir="/workspaces/udop/i-Code-Doc/IA4RobotFramework/Web/frontend/data/to_udop", cache_dir=cache_dir)
    assert isinstance(dataset, DatasetDict), "Please provide a dataset dict"
    example = dataset["train"][0]
    # Run pipeline
    image = PIL.Image.open(io.BytesIO(base64.b64decode(example['screenshot']))).convert('RGB')
    image.save("test.png")
    print(example['instruction_history'])
    instruction = UdopExampleToInstruction(
                tokenizer, image.size
            ).build(example['instruction_history'])
    print(instruction)
    with open("test.txt", "w") as f:
        f.write(instruction)
    prediction = udop({"image":image, "instruction":instruction})
    print(f'Label: {example["step"]["name"]} {example["step"]["args"]["bbox"]}')
    print(f'Prediction {prediction}')
