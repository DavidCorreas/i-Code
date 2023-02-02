# Load training_args.bin file
import torch
import json
from collections import OrderedDict


model_dir = "/workspaces/udop/i-Code-Doc/model"
download_dir = "/workspaces/udop/i-Code-Doc/model/UdopUnimodel-Large-224"

# Modify the keys of the model
udop_download = torch.load(f'{download_dir}/pytorch_model.bin')
udop_model = OrderedDict({k.replace('model.', ''): v for k, v in udop_download.items()})
for i in range(len(udop_model)):
    assert list(udop_model.keys())[i] == list(udop_download.keys())[i].replace('model.', '')
torch.save(udop_model, f'{model_dir}/pytorch_model.bin')

# Modify the config architecture name.
with open(f'{model_dir}/config.json', 'r') as f:
    config = json.load(f)
# Modify the architecture name
config['architectures'] = ['UdopUnimodelForConditionalGeneration']
config['vocab_size'] = 33201  # Due to the special tokens. Detected in the .bin model.encoder.embed_tokens.weight = torch.Size([33201, 1024]) instead of 32128
with open(f'{model_dir}/config.json', 'w') as f:
    json.dump(config, f)


# Generate tokenizer
print('Go to the following link to know how to generate the tokenizer: https://www.notion.so/UDOP-Ejecuci-n-065fa072df274e1eb088f019248b2318#31f54868a3bc4ab2a6c8abf0a783aaf3')
