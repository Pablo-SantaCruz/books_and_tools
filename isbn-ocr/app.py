from flask import Flask, request, jsonify
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

@app.route("/ocr", methods=["POST"])
def ocr_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img)
    text = pytesseract.image_to_string(img_np)
    # Aquí puedes agregar lógica para extraer el ISBN del texto
    return jsonify({"text": text})

@app.route("/")
def hello():
    return "Hola desde isbn-ocr"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)