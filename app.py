"""
=============================================================
  PNEUMONIA DETECTION — FLASK WEB APPLICATION
  Run: python app.py
  Open: http://localhost:5000
=============================================================

HOW THIS WORKS (Beginner Explanation)
--------------------------------------
Flask is a lightweight Python web framework. Think of it as a
small web server that:
  1. Serves the HTML page to your browser
  2. Receives the uploaded X-ray image
  3. Runs it through the trained CNN model
  4. Returns the prediction result back to the browser

The whole thing runs on YOUR machine — no internet required!
"""

import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_PATH = r"F:\Lock In\Skripsi\Script\pneumonia_model.keras"   # <-- path to your trained model
IMG_SIZE   = (224, 224)
THRESHOLD  = 0.5                      # probability threshold for classification

# Labels (must match what was used during training)
CLASS_LABELS = {0: "NORMAL", 1: "PNEUMONIA"}

# ─────────────────────────────────────────────
# INITIALIZE FLASK APP & LOAD MODEL
# ─────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

print("🧠 Loading trained model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Could not load model: {e}")
    print("   → Train the model first by running: python train.py")
    model = None

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Prepare an uploaded image for the CNN model.
    Steps:
      1. Open image from bytes
      2. Convert to RGB (in case it's grayscale or RGBA)
      3. Resize to 224×224 (what MobileNetV2 expects)
      4. Normalize pixels to [0, 1]
      5. Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


def predict(image_bytes: bytes) -> dict:
    """
    Run the model and return a prediction dictionary.
    Returns:
        {
          "label"      : "PNEUMONIA" or "NORMAL",
          "confidence" : 0.95,           (probability 0–1)
          "probability_normal"   : 0.05,
          "probability_pneumonia": 0.95,
          "message"    : "...",
          "risk_level" : "High" / "Low"
        }
    """
    if model is None:
        raise RuntimeError("Model is not loaded. Please run train.py first.")

    img_array = preprocess_image(image_bytes)

    # model.predict returns a probability between 0 and 1
    # Values close to 1 → PNEUMONIA, close to 0 → NORMAL
    prob_pneumonia = float(model.predict(img_array, verbose=0)[0][0])
    prob_normal    = 1.0 - prob_pneumonia

    label      = "PNEUMONIA" if prob_pneumonia >= THRESHOLD else "NORMAL"
    confidence = prob_pneumonia if label == "PNEUMONIA" else prob_normal

    risk_level = "High" if prob_pneumonia >= 0.7 else \
                 "Moderate" if prob_pneumonia >= 0.4 else "Low"

    messages = {
        "PNEUMONIA": (
            "The model detects signs consistent with pneumonia. "
            "Please consult a qualified medical professional immediately."
        ),
        "NORMAL": (
            "No signs of pneumonia detected. "
            "However, this AI tool is not a substitute for professional medical advice."
        ),
    }

    return {
        "label"               : label,
        "confidence"          : round(confidence * 100, 2),
        "probability_normal"  : round(prob_normal * 100, 2),
        "probability_pneumonia": round(prob_pneumonia * 100, 2),
        "message"             : messages[label],
        "risk_level"          : risk_level,
    }

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Endpoint that receives an uploaded image and returns a JSON prediction.
    Called by the JavaScript in the frontend via fetch().
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # Check file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type: {ext}. Use JPG, PNG, or BMP."}), 400

    try:
        image_bytes = file.read()
        result      = predict(image_bytes)

        # Encode the image as base64 so the frontend can display it
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        result["image_data"] = f"data:image/{ext.strip('.')};base64,{b64_image}"

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Simple health check endpoint."""
    return jsonify({
        "status" : "ok",
        "model_loaded": model is not None,
        "model_path"  : MODEL_PATH,
    })


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🌐 Starting Flask server...")
    print("   Open your browser at: http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
