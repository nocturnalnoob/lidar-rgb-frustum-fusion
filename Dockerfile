# CPU-only image for the LiDAR + RGB fusion web app.
FROM python:3.12-slim

# System libs needed by OpenCV / matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (smaller, no CUDA), then the rest.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Fetch sample frames + YOLO weights at build time so the container is self-contained.
RUN python scripts/download_sample.py --out data/kitti || true

# Hugging Face Spaces (and most PaaS) inject $PORT; default to 7860 for HF.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
