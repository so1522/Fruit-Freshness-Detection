import os
import sys
import base64
import time
from flask import Flask, request, jsonify, send_from_directory

# Ensure project root directory and backend directory are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from backend.model_loader import get_model_loader
except ModuleNotFoundError:
    from model_loader import get_model_loader

# Initialize Flask app pointing to frontend folder
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# Initialize model loader
model_loader = get_model_loader()


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "model_name": model_loader.best_model_name,
        "classes": list(model_loader.label_encoder.classes_),
        "version": "2.0.0",
        "description": "Fruit Freshness Detection ML API"
    })


@app.route("/api/samples", methods=["GET"])
def get_sample_images():
    sample_files = [
        {"id": "fa.jpeg", "name": "Fresh Apple Sample", "category": "Fresh Apple", "filename": "fa.jpeg"},
        {"id": "fo.jpeg", "name": "Fresh Orange Sample", "category": "Fresh Orange", "filename": "fo.jpeg"},
        {"id": "ro.jpeg", "name": "Rotten Orange Sample", "category": "Rotten Orange", "filename": "ro.jpeg"},
        {"id": "bb.jpeg", "name": "Banana Sample", "category": "Banana", "filename": "bb.jpeg"},
        {"id": "ra.jpeg", "name": "Rotten Apple Sample", "category": "Rotten Apple", "filename": "ra.jpeg"},
        {"id": "ro1jpeg.jpeg", "name": "Rotten Produce Sample", "category": "Rotten Fruit", "filename": "ro1jpeg.jpeg"}
    ]
    
    existing_samples = []
    for sample in sample_files:
        if os.path.exists(os.path.join(ROOT_DIR, sample["filename"])):
            existing_samples.append(sample)

    return jsonify({"samples": existing_samples})


@app.route("/api/sample-image/<filename>", methods=["GET"])
def serve_sample_image(filename):
    return send_from_directory(ROOT_DIR, filename)


@app.route("/api/predict", methods=["POST"])
def predict_fruit():
    start_time = time.time()
    img_bytes = None

    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            img_bytes = file.read()

    if img_bytes is None and request.is_json:
        data = request.get_json()
        image_data = data.get("image")
        if image_data:
            if "," in image_data:
                image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)

    if img_bytes is None and request.is_json:
        data = request.get_json()
        sample_name = data.get("sample")
        if sample_name:
            sample_path = os.path.join(ROOT_DIR, sample_name)
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    img_bytes = f.read()

    if not img_bytes:
        return jsonify({
            "error": "No valid image provided. Please upload an image file or base64 data."
        }), 400

    try:
        result = model_loader.predict_bytes(img_bytes)
        processing_time = round(time.time() - start_time, 3)
        result["processing_time_sec"] = processing_time
        return jsonify(result)
    except Exception as e:
        print(f"[API Error] {e}")
        return jsonify({
            "error": f"Failed to process image: {str(e)}"
        }), 500


@app.route("/<path:path>")
def serve_static(path):
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    print(f"Starting Fruit Freshness Detection App at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
