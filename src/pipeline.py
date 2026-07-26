"""
End-to-end fusion pipeline used by both the CLI (scripts/run_fusion.py) and the
Flask web app. Loads a KITTI frame, runs 2D detection + LiDAR frustum fusion,
renders every visualization, and evaluates against ground truth when available.
"""

import os
import cv2

from src.data.kitti_loader import KittiLoader
from src.detection.yolo_detector import YoloDetector
from src.fusion.frustum_fusion import fuse_frame
from src.fusion.metrics import evaluate_frame
from src.calibration.project_lidar import project_velo_to_image
from src.visualization.draw import (
    draw_2d_detections,
    draw_3d_detections,
    render_bev,
)
from src.visualization import draw as _draw_mod  # noqa: F401 (ensures Agg backend set)
from src.calibration.project_lidar import overlay_lidar_on_image


class FusionPipeline:
    """Holds the (lazily-loaded) detector and runs frames on demand."""

    def __init__(self, data_dir='data/kitti', weights='yolov8s.pt',
                 conf=0.35, device='cpu', head='geometric',
                 frustum_weights='models/frustum_pointnet.pt'):
        self.loader = KittiLoader(data_dir, split='training')
        self.weights = weights
        self.conf = conf
        self.device = device
        self.head = head                         # 'geometric' or 'learned'
        self.frustum_weights = frustum_weights
        self._detector = None
        self._frustum_model = None

    @property
    def detector(self):
        if self._detector is None:
            self._detector = YoloDetector(self.weights, conf=self.conf,
                                          device=self.device)
        return self._detector

    @property
    def frustum_model(self):
        if self._frustum_model is None:
            from src.fusion.frustum_pointnet import load_frustum_model
            self._frustum_model = load_frustum_model(self.frustum_weights)
        return self._frustum_model

    def list_frames(self):
        return self.loader.list_frames()

    def run(self, idx, render=True, head=None):
        """
        Process one frame. Returns a dict with detections, metrics and (if
        render=True) BGR image arrays for each visualization stage.

        `head` overrides the instance default ('geometric' or 'learned').
        """
        head = head or self.head
        calib = self.loader.get_calib(idx)
        image = self.loader.get_image(idx)
        lidar = self.loader.get_lidar(idx)

        detections_2d = self.detector.detect(image)
        model = self.frustum_model if head == 'learned' else None
        detections_3d = fuse_frame(lidar, calib, detections_2d,
                                   head=head, frustum_model=model)

        gt_objects = self.loader.get_labels(idx) if self.loader.has_labels(idx) else []
        metrics = (evaluate_frame(detections_3d, gt_objects, calib)
                   if gt_objects else None)

        result = {
            'idx': idx,
            'head': head,
            'image_shape': image.shape,
            'n_lidar': int(lidar.shape[0]),
            'detections_2d': detections_2d,
            'detections_3d': detections_3d,
            'gt_objects': gt_objects,
            'metrics': metrics,
        }

        if render:
            pts_2d, pts_cam, _ = project_velo_to_image(lidar, calib)
            result['images'] = {
                'rgb': image,
                'lidar_projection': overlay_lidar_on_image(
                    image, pts_2d, pts_cam[:, 2]),
                'detections_2d': draw_2d_detections(image, detections_2d),
                'detections_3d': draw_3d_detections(image, detections_3d, calib),
                'bev': render_bev(lidar, detections_3d, gt_objects, calib),
            }
        return result

    @staticmethod
    def save_images(images, out_dir, prefix):
        os.makedirs(out_dir, exist_ok=True)
        paths = {}
        for name, img in images.items():
            path = os.path.join(out_dir, f"{prefix}_{name}.png")
            cv2.imwrite(path, img)
            paths[name] = path
        return paths


def detections_to_kitti_lines(detections_3d):
    """Serialize 3D detections to KITTI label format strings."""
    lines = []
    for d in detections_3d:
        h, w, l = d['dimensions']
        x, y, z = d['location']
        x1, y1, x2, y2 = d['bbox']
        lines.append(
            f"{d['kitti_class']} 0.00 0 -10 "
            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            f"{h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} "
            f"{d['rotation_y']:.2f} {d['score']:.3f}"
        )
    return lines
