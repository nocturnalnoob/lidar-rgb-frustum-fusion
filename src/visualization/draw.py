"""
Rendering helpers: 2D boxes, projected 3D boxes on the image, and a bird's-eye
view (BEV) of the LiDAR scene with predicted and ground-truth boxes.

All functions are headless-safe (matplotlib uses the Agg backend) so they work
on a server with no display.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.detection.yolo_detector import KITTI_COLORS
from src.calibration.project_lidar import (
    velo_to_rect,
    project_rect_to_image,
    compute_box_3d_corners_rect,
    rect_to_velo,
)

# Edges of a box given 8 corners ordered bottom(0-3) then top(4-7).
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),        # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),        # top face
    (0, 4), (1, 5), (2, 6), (3, 7),        # vertical pillars
]


def draw_2d_detections(image, detections):
    """Draw YOLO 2D boxes with class + score labels."""
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox'].astype(int)
        color = KITTI_COLORS.get(det['kitti_class'], (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det['kitti_class']} {det['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _project_corners_velo(corners_velo, calib):
    rect = velo_to_rect(corners_velo, calib)
    pts_2d, _ = project_rect_to_image(rect, calib)
    return pts_2d


def draw_3d_detections(image, detections_3d, calib):
    """Draw projected 3D boxes (wireframe) plus a class/depth label."""
    img = image.copy()
    for det in detections_3d:
        corners_2d = _project_corners_velo(det['corners_velo'], calib).astype(int)
        color = KITTI_COLORS.get(det['kitti_class'], (200, 200, 200))
        for i, j in _BOX_EDGES:
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]),
                     color, 2, cv2.LINE_AA)
        # Front face highlighted by a filled translucent quad.
        top = corners_2d[4]
        label = f"{det['kitti_class']} {det['depth']:.1f}m"
        cv2.putText(img, label, (top[0], max(top[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return img


def _bev_corners_from_label(obj, calib):
    """Ground-truth KITTI object -> 4 BEV corners (x, y) in the Velodyne frame."""
    corners_rect = compute_box_3d_corners_rect(
        obj['dimensions'], obj['location'], obj['rotation_y'])
    corners_velo = rect_to_velo(corners_rect, calib)
    return corners_velo[:4, :2]  # bottom face is enough for BEV


def render_bev(lidar, detections_3d, gt_objects=None, calib=None,
               x_range=(0, 60), y_range=(-25, 25), figsize=(7, 7)):
    """
    Render a bird's-eye view: LiDAR points (grey), predicted boxes (colored),
    and optional ground-truth boxes (dashed white). Returns a BGR image array.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    pts = lidar
    m = ((pts[:, 0] > x_range[0]) & (pts[:, 0] < x_range[1]) &
         (pts[:, 1] > y_range[0]) & (pts[:, 1] < y_range[1]))
    p = pts[m]
    ax.scatter(p[:, 1], p[:, 0], s=0.25, c=p[:, 2], cmap='viridis',
               alpha=0.55, linewidths=0)

    # Predicted boxes.
    for det in detections_3d:
        bev = det['corners_velo'][:4, :2]
        poly = np.vstack([bev, bev[0]])
        color = np.array(KITTI_COLORS.get(det['kitti_class'], (200, 200, 200)))[::-1] / 255.0
        ax.plot(poly[:, 1], poly[:, 0], '-', color=color, linewidth=2)
        c = bev.mean(axis=0)
        ax.text(c[1], c[0], det['kitti_class'][0], color=color,
                fontsize=9, ha='center', va='center', fontweight='bold')

    # Ground-truth boxes.
    if gt_objects and calib is not None:
        for obj in gt_objects:
            if obj['type'] in ('DontCare',):
                continue
            bev = _bev_corners_from_label(obj, calib)
            poly = np.vstack([bev, bev[0]])
            ax.plot(poly[:, 1], poly[:, 0], '--', color='white',
                    linewidth=1.2, alpha=0.7)

    # Sensor origin + range rings.
    ax.plot(0, 0, marker='^', color='red', markersize=9)
    for r in (10, 20, 30, 40, 50):
        circ = plt.Circle((0, 0), r, color='#30363d', fill=False, linewidth=0.6)
        ax.add_patch(circ)

    ax.set_xlim(y_range[1], y_range[0])  # flip so +y(left) is on the left
    ax.set_ylim(x_range[0], x_range[1])
    ax.set_xlabel('y (left) [m]', color='#8b949e')
    ax.set_ylabel('x (forward) [m]', color='#8b949e')
    ax.set_title('Bird\'s-Eye View — fused 3D boxes', color='#c9d1d9')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.set_aspect('equal')
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img_bgr
