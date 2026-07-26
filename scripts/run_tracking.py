"""
Multi-object tracking demo on a KITTI tracking sequence.

For each frame: YOLOv8 2D detection -> LiDAR frustum fusion -> 3D boxes -> SORT
tracker (stable IDs). Renders an image+BEV walkthrough GIF where each track keeps
its color/ID across frames, and reports track-count / track-length statistics.

Usage:
    python scripts/run_tracking.py --seq 0000 --data_dir data/kitti_tracking \
        --out docs/assets/tracking.gif --max_frames 90
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from src.data.tracking_loader import TrackingLoader
from src.detection.yolo_detector import YoloDetector
from src.fusion.frustum_fusion import fuse_frame
from src.tracking.sort import Sort, Track
from src.calibration.project_lidar import velo_to_rect, project_rect_to_image

# Distinct BGR colors cycled by track id.
PALETTE = [(66, 212, 244), (113, 204, 46), (60, 76, 231), (219, 152, 52),
           (182, 89, 155), (34, 126, 230), (156, 188, 26), (43, 57, 192),
           (185, 128, 41), (96, 174, 39), (0, 165, 255), (214, 112, 218)]
_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def color_for(tid):
    return PALETTE[tid % len(PALETTE)]


def draw_tracks_image(image, tracks, calib):
    img = image.copy()
    for t in tracks:
        rect = velo_to_rect(t.det['corners_velo'], calib)
        pts2d, _ = project_rect_to_image(rect, calib)
        pts2d = pts2d.astype(int)
        c = color_for(t.id)
        for i, j in _BOX_EDGES:
            cv2.line(img, tuple(pts2d[i]), tuple(pts2d[j]), c, 2, cv2.LINE_AA)
        top = pts2d[4]
        cv2.putText(img, f"ID{t.id} {t.kitti_class}", (top[0], max(top[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
    return img


def render_tracks_bev(lidar, tracks, x_range=(0, 60), y_range=(-25, 25)):
    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=90)
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
    p = lidar[::3]                       # subsample points to keep the GIF small
    m = ((p[:, 0] > x_range[0]) & (p[:, 0] < x_range[1]) &
         (p[:, 1] > y_range[0]) & (p[:, 1] < y_range[1]))
    q = p[m]
    ax.scatter(q[:, 1], q[:, 0], s=0.2, c='#30363d', alpha=0.5, linewidths=0)
    for t in tracks:
        bev = t.det['corners_velo'][:4, :2]
        poly = np.vstack([bev, bev[0]])
        c = np.array(color_for(t.id))[::-1] / 255.0
        ax.plot(poly[:, 1], poly[:, 0], '-', color=c, linewidth=2)
        cen = bev.mean(axis=0)
        ax.text(cen[1], cen[0], str(t.id), color=c, fontsize=9, ha='center',
                va='center', fontweight='bold')
        if len(t.history) > 1:
            h = np.array(t.history)
            ax.plot(h[:, 1], h[:, 0], '-', color=c, linewidth=1, alpha=0.5)
    ax.plot(0, 0, marker='^', color='red', markersize=8)
    ax.set_xlim(y_range[1], y_range[0]); ax.set_ylim(x_range[0], x_range[1])
    ax.set_aspect('equal'); ax.axis('off')
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    out = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return out


def compose(img_bgr, bev_bgr, width=680):
    scale = width / img_bgr.shape[1]
    top = cv2.resize(img_bgr, (width, int(img_bgr.shape[0] * scale)))
    bh = min(bev_bgr.shape[0], 460)
    bscale = bh / bev_bgr.shape[0]
    bev = cv2.resize(bev_bgr, (int(bev_bgr.shape[1] * bscale), bh))
    canvas = np.full((top.shape[0] + bev.shape[0], width, 3), (13, 17, 23), np.uint8)
    canvas[:top.shape[0]] = top
    x0 = (width - bev.shape[1]) // 2
    canvas[top.shape[0]:top.shape[0] + bev.shape[0], x0:x0 + bev.shape[1]] = bev
    return canvas


def main():
    ap = argparse.ArgumentParser(description="MOT demo on a KITTI tracking sequence")
    ap.add_argument('--data_dir', default='data/kitti_tracking')
    ap.add_argument('--seq', default='0000')
    ap.add_argument('--out', default='docs/assets/tracking.gif')
    ap.add_argument('--weights', default='yolov8s.pt')
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--max_frames', type=int, default=70)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--max_age', type=int, default=8)
    ap.add_argument('--min_hits', type=int, default=3)
    args = ap.parse_args()

    loader = TrackingLoader(args.data_dir, seq=args.seq)
    detector = YoloDetector(args.weights, conf=args.conf)
    Sort.reset_ids(); Track._next_id = 1
    tracker = Sort(iou_thresh=0.1, max_age=args.max_age, min_hits=args.min_hits)

    frames = loader.list_frames()[:args.max_frames:args.stride]
    calib = loader.get_calib()
    gif_frames = []
    track_lengths = {}   # id -> frame count
    for n, idx in enumerate(frames, 1):
        image = loader.get_image(idx)
        lidar = loader.get_lidar(idx)
        dets2d = detector.detect(image)
        dets3d = fuse_frame(lidar, calib, dets2d)
        active = tracker.update(dets3d)
        for t in active:
            track_lengths[t.id] = track_lengths.get(t.id, 0) + 1
        frame_img = compose(draw_tracks_image(image, active, calib),
                            render_tracks_bev(lidar, active))
        gif_frames.append(Image.fromarray(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)))
        if n % 20 == 0 or n == len(frames):
            print(f"  frame {n}/{len(frames)}: {len(active)} active tracks, "
                  f"{len(track_lengths)} total IDs")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    gif_frames[0].save(args.out, save_all=True, append_images=gif_frames[1:],
                       duration=120, loop=0, optimize=True)

    lengths = np.array(list(track_lengths.values()))
    print(f"\nSequence {args.seq}: {len(frames)} frames processed.")
    print(f"  unique track IDs: {len(track_lengths)}")
    print(f"  track length (frames): median={int(np.median(lengths))} "
          f"mean={lengths.mean():.1f} max={lengths.max()}")
    print(f"  tracks lasting >=10 frames: {(lengths >= 10).sum()}")
    print(f"Saved GIF -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
