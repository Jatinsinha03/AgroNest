
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
from chat import get_response  # Assuming get_response is the function from your chat module

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model using TFSMLayer
MODEL = tf.keras.layers.TFSMLayer("models/1", call_endpoint='serving_default')

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.get("/")
def index_get():
    return render_template("base.html")

def read_file_as_image(file) -> np.ndarray:
    image = Image.open(file)
    image = image.resize((256, 256))  # Resize image to the input size of the model
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image

@tf.function
def predict_image(image):
    return MODEL(image)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file provided"}), 400

    image = read_file_as_image(file.stream)  # Read the file in binary mode
    img_batch = np.expand_dims(image, 0)

    predictions = predict_image(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions['dense_1'][0])]
    confidence = np.max(predictions['dense_1'][0]) * 100

    return jsonify({
        'class': predicted_class,
        'confidence': float(confidence)
    })

@app.route("/chat", methods=['POST'])
def chat():
    text = request.get_json().get("message")
    if text is None:
        return jsonify({"error": "No Message provided"}), 400
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == '__main__':
    app.run(debug=True)
