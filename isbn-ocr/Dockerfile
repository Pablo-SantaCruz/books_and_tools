FROM python:3.11-slim

# Instala dependencias del sistema, incluyendo libgl1
RUN apt-get update && \
    apt-get install -y tesseract-ocr libgl1 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
