"""
Build an animated GIF that walks through the fusion pipeline stages for a frame:
RGB -> LiDAR projection -> 2D detection -> 3D fusion -> bird's-eye view.

Usage:
    python scripts/make_demo_gif.py --idx 0 --out docs/assets/demo.gif
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from PIL import Image

from src.pipeline import FusionPipeline

STAGES = [
    ('rgb', 'RGB camera'),
    ('lidar_projection', 'LiDAR projected onto image (depth-colored)'),
    ('detections_2d', 'YOLOv8 2D detection'),
    ('detections_3d', 'Fused 3D boxes'),
    ('bev', "Bird's-eye view (green=pred, dashed=GT)"),
]
CANVAS_W = 1000
CANVAS_H = 460          # content area; wide KITTI images and the square BEV both fit
BAR = 46               # caption bar height
BG = (13, 17, 23)      # dark background (BGR)
FG = (230, 237, 246)


def _frame_canvas(img_bgr, caption):
    """Letterbox a BGR image (centered) into a fixed canvas with a caption bar."""
    h, w = img_bgr.shape[:2]
    scale = min(CANVAS_W / w, CANVAS_H / h)
    nw, nh = int(w * scale), int(h * scale)
    img = cv2.resize(img_bgr, (nw, nh))
    canvas = np.full((CANVAS_H + BAR, CANVAS_W, 3), BG, dtype=np.uint8)
    x0 = (CANVAS_W - nw) // 2
    y0 = BAR + (CANVAS_H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = img
    cv2.putText(canvas, caption, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                FG[::-1], 2, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Make a pipeline walkthrough GIF")
    ap.add_argument('--data_dir', default='data/kitti')
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--out', default='docs/assets/demo.gif')
    ap.add_argument('--ms', type=int, default=1300, help='ms per stage')
    args = ap.parse_args()

    pipe = FusionPipeline(args.data_dir, conf=0.35)
    res = pipe.run(args.idx, render=True)
    images = res['images']

    # All canvases share a fixed size, so the GIF playback stays stable.
    pil_frames = [
        Image.fromarray(cv2.cvtColor(_frame_canvas(images[k], cap), cv2.COLOR_BGR2RGB))
        for k, cap in STAGES
    ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pil_frames[0].save(
        args.out, save_all=True, append_images=pil_frames[1:],
        duration=args.ms, loop=0, optimize=True)
    print(f"Saved demo GIF -> {args.out} ({len(pil_frames)} stages, "
          f"{os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
