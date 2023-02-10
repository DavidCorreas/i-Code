
# import flask
import numpy as np
import flask
from PIL import Image
from utils import initialize_ocr, ocr

# Initialize the OCR object due to memory leak
ocr_object  = initialize_ocr()

# Create a Flask app
app = flask.Flask(__name__)

@app.route('/health')
def health():
    return 'OK', 200

# Create a route to read an image and return the text with paddle
@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    # Read the image from the request
    image = flask.request.files["image"]

    # Convert the _io.BytesIO to a PIL image
    image = Image.open(image.stream).convert("RGB")
    np_image = np.array(image)

    # Create the OCR object, get the language from the request
    # check if exists, if not, use english
    result = ocr(ocr_object, np_image)
    del np_image
    del image

    # Return the text
    return flask.jsonify({"result": result})


# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
