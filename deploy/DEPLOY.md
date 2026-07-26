# Deploying the fusion web app

The app is a single Flask service; it reads its port from `$PORT` (default 5000
locally, 7860 in the container). Two easy hosting options:

## Option A — Docker (works anywhere)

```bash
docker build -t lidar-rgb-fusion .
docker run -p 7860:7860 lidar-rgb-fusion
# open http://localhost:7860
```

The image installs CPU-only PyTorch, fetches the 3 sample KITTI frames, and
pulls the YOLOv8n weights at build time, so the container runs offline.

## Option B — Hugging Face Spaces (free, gives you a public URL)

1. Create a new **Space** → SDK: **Docker**.
2. Push this repo to the Space (it already contains a `Dockerfile`).
3. Add this front-matter to the **top of the Space's `README.md`** so Spaces
   knows which port to expose:

   ```yaml
   ---
   title: Multi-Sensor LiDAR x RGB Fusion
   emoji: 🚗
   colorFrom: green
   colorTo: blue
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

4. The Space builds the Docker image and serves the app at
   `https://huggingface.co/spaces/<your-username>/<space-name>`.

> Tip for your résumé: link the live Space directly. A recruiter clicking a
> working demo is worth more than any screenshot.
