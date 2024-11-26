from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS


app = Flask(__name__)

CORS(app)

# Directory to save uploaded images
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = tf.keras.models.load_model("trained_model.h5")

# Prediction function
def model_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# API route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    print(f"Received file: {file.filename}")
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Perform prediction
    result_index = model_prediction(filepath)
    # Read labels from file
    with open("labels.txt") as f:
        labels = [line.strip() for line in f]
    
    # Return prediction result
    os.remove(filepath)  # Clean up the uploaded file
    return jsonify({"prediction": labels[result_index]})

# API route for project details
@app.route("/about", methods=["GET"])
def about():
    details = {
        "header": "FRUITS & VEGETABLES RECOGNITION SYSTEM",
        "about_dataset": [
            "fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.",
            "vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, "
            "lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet "
            "potato, paprika, jalepeño, ginger, garlic, peas, eggplant."
        ],
        "content": [
            "This dataset contains three folders:",
            "1. train (100 images each)",
            "2. test (10 images each)",
            "3. validation (10 images each)"
        ]
    }
    return jsonify(details)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
