from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = "mulberry_leaf_disease_Identification.keras"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Model file not found at {MODEL_PATH}")

# Class labels — must match your training order
CLASS_NAMES = ["Disease Free", "Potential Leaf Rust", "Potential Leaf Spot"]

# Recommendations for each class
RECOMMENDATIONS = {
    "Disease Free": [
        "✅ Leaf appears healthy. Continue regular monitoring.",
        "💧 Maintain proper irrigation and avoid waterlogging.",
        "🌿 Apply balanced NPK fertilizer every 2–3 weeks.",
        "🔍 Inspect plants weekly to catch early signs of disease.",
    ],
    "Potential Leaf Rust": [
        "⚠️ Rust detected! Remove and destroy infected leaves immediately.",
        "🧪 Apply copper-based fungicide (e.g., Mancozeb) every 7–10 days.",
        "💨 Improve air circulation by pruning dense foliage.",
        "🚫 Avoid overhead watering — water at the base of plants.",
        "🔄 Rotate crops next season to break the disease cycle.",
    ],
    "Potential Leaf Spot": [
        "⚠️ Leaf spot identified! Remove heavily infected leaves.",
        "🧪 Spray with Chlorothalonil or Carbendazim fungicide.",
        "🌱 Ensure proper plant spacing for better airflow.",
        "💧 Water early morning so leaves dry during the day.",
        "🧹 Clean up fallen leaf debris around the plant base.",
    ],
}

SEVERITY = {
    "Disease Free": "healthy",
    "Potential Leaf Rust": "warning",
    "Potential Leaf Spot": "danger",
}

IMG_SIZE = (128, 128)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        predicted_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100

        all_confidences = {
            CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "all_confidences": all_confidences,
            "recommendations": RECOMMENDATIONS[predicted_class],
            "severity": SEVERITY[predicted_class],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5000)
