import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import base64
import io
import logging
logging.basicConfig(level=logging.INFO)

# Declare a Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'anemalyze_model.h5'
model = load_model("anemalyze_model.h5")
print(model.summary())

# Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}. Check http://127.0.0.1:5000/")

def model_predict(img, model):
    img = img.resize((224, 224))  # Resize image to match model input size

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # Normalize the input for MobileNetV2

    preds = model.predict(x)
    return preds

def base64_to_pil(img_base64):
    """Convert base64 image data to PIL image."""
    img_data = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_data))
    return img

@app.route('/', methods=['GET'])
def index():
    return "Aplikasi Anemalyze Berjalan"


@app.route('/predict', methods=['POST'])
def predict():
    logging.info(f"Received request with method: {request.method}")
    logging.info(f"Request data: {request.json}")
    if request.method == 'POST':
        try:
            img_base64 = request.json.get('image')
            if not img_base64:
                return jsonify({"error": "No image provided"}), 400

            # Convert base64 image to PIL image
            img = base64_to_pil(img_base64)

            # Make prediction
            preds = model_predict(img, model)

            # Extract the probabilities for each class
            anemia_prob = preds[0][0]
            non_anemia_prob = preds[0][1]

            # Determine the class with the highest probability
            if anemia_prob > non_anemia_prob:
                result = "Anemia"
                probability = anemia_prob
            else:
                result = "Non-Anemia"
                probability = non_anemia_prob

            # Serialize the result
            return jsonify(result=result, probability="{:.3f}".format(probability))
        except Exception as e:
            # Log and return error
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid request method"}), 405

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
