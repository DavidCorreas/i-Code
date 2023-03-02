# Create a flask app with a post route that recives a image and a prompt and returns a action prediction with a huggingface model
import flask
import io
import base64
import PIL.Image
from transformers import HfArgumentParser
from dataclasses import dataclass
from core.datasets.robotframework import UdopExampleToInstruction
from core.models import UdopUnimodelForConditionalGeneration, UdopTokenizer, UdopPipeline
from src.qact.data_structure import PromptStep


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@dataclass
class ModelConfig:
    model_name_or_path: str = "/workspaces/udop/i-Code-Doc/finetune_robotframework/checkpoint-1500"

@dataclass
class FlaskConfig:
    host: str = "0.0.0.0"
    port: int = 5000


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image and prompt
    image = flask.request.files['image']
    prompt = flask.request.form['instruction']

    # Parse image to pil image
    image = PIL.Image.open(image.stream).convert("RGB")
    prediction = udop_pipeline({"image":image, "instruction":prompt})

    # Return the action
    return flask.jsonify({"action": prediction})

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    json = flask.request.get_json()

    # Parse the image from base64 to pil imag
    image = json['image']
    image = PIL.Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
    
    # Parse the instruction history to a prompt
    instruction_history_d: list[dict] = json['instruction_history']
    instruction_history: list[PromptStep] = [PromptStep.from_dict(step) for step in instruction_history_d]
    print(instruction_history)
    
    prompt = UdopExampleToInstruction(
                tokenizer, image.size
            ).build(instruction_history)
    print(prompt)
    prediction = udop_pipeline({"image":image, "instruction":prompt})

    # Return the action
    return flask.jsonify({"action": prediction})

# Run the app
if __name__ == "__main__":
    # Parse the config with argparse
    parser = HfArgumentParser((FlaskConfig, ModelConfig))
    flask_config, model_config = parser.parse_args_into_dataclasses()
    
    # Init
    model = UdopUnimodelForConditionalGeneration.from_pretrained(model_config.model_name_or_path)
    tokenizer = UdopTokenizer.from_pretrained(model_config.model_name_or_path)
    global udop_pipeline
    udop_pipeline = UdopPipeline(model=model, tokenizer=tokenizer)

    # Run the app
    app.run(host=flask_config.host, port=flask_config.port)

