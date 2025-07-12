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

    # Escala de grises y mejora de contraste
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    img_gray_eq = cv2.equalizeHist(img_gray)

    # Umbral fijo
    _, img_thresh = cv2.threshold(img_gray_eq, 150, 255, cv2.THRESH_BINARY)

    # Umbral adaptativo
    img_adapt = cv2.adaptiveThreshold(
        img_gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )

    # Probar varios modos PSM y elegir el mejor resultado
    best_text = ""
    best_len = 0
    for psm in [3, 4, 6]:
        config = f'--psm {psm}'
        for img_proc in [img_thresh, img_adapt]:
            text = pytesseract.image_to_string(img_proc, lang='spa', config=config)
            if len(text) > best_len:
                best_text = text
                best_len = len(text)

    return jsonify({"text": best_text})

@app.route("/")
def hello():
    return "Hola desde isbn-ocr"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)