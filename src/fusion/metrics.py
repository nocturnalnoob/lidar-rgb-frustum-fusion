"""
Lightweight bird's-eye-view (BEV) evaluation against KITTI ground truth.

We compute polygon IoU between predicted and ground-truth boxes in the BEV
plane (Velodyne x-y), then greedily match per class to report precision,
recall and mean matched IoU. This is a pragmatic stand-in for the official
KITTI AP and is plenty to demonstrate the fusion works.
"""

import cv2
import numpy as np

from src.calibration.project_lidar import (
    compute_box_3d_corners_rect,
    rect_to_velo,
)


def _poly_iou(poly_a, poly_b):
    """IoU of two convex BEV polygons given as (4,2) arrays."""
    a = poly_a.astype(np.float32)
    b = poly_b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    if inter <= 0:
        return 0.0
    area_a = cv2.contourArea(a)
    area_b = cv2.contourArea(b)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _gt_bev_corners(obj, calib):
    corners_rect = compute_box_3d_corners_rect(
        obj['dimensions'], obj['location'], obj['rotation_y'])
    return rect_to_velo(corners_rect, calib)[:4, :2]


def evaluate_frame(detections_3d, gt_objects, calib, iou_thresh=0.25):
    """
    Match predictions to ground truth in BEV and return a metrics dict with
    per-class and overall precision / recall / mean IoU plus the matched pairs.
    """
    classes = ('Car', 'Pedestrian', 'Cyclist')
    gt = [o for o in gt_objects if o['type'] in classes]

    preds = list(detections_3d)
    pred_polys = [d['corners_velo'][:4, :2] for d in preds]
    gt_polys = [_gt_bev_corners(o, calib) for o in gt]

    matched_gt = set()
    matches = []  # (pred_idx, gt_idx, iou)
    # Greedy: highest-score predictions claim the best free GT of same class.
    order = sorted(range(len(preds)), key=lambda i: -preds[i]['score'])
    for pi in order:
        best_iou, best_gi = 0.0, -1
        for gi, obj in enumerate(gt):
            if gi in matched_gt or obj['type'] != preds[pi]['kitti_class']:
                continue
            iou = _poly_iou(pred_polys[pi], gt_polys[gi])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0 and best_iou >= iou_thresh:
            matched_gt.add(best_gi)
            matches.append((pi, best_gi, best_iou))

    def _stats(subset_classes):
        n_pred = sum(1 for d in preds if d['kitti_class'] in subset_classes)
        n_gt = sum(1 for o in gt if o['type'] in subset_classes)
        m = [iou for pi, gi, iou in matches
             if preds[pi]['kitti_class'] in subset_classes]
        tp = len(m)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gt if n_gt else 0.0
        mean_iou = float(np.mean(m)) if m else 0.0
        return {
            'n_pred': n_pred, 'n_gt': n_gt, 'tp': tp,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'mean_iou': round(mean_iou, 3),
        }

    per_class = {c: _stats({c}) for c in classes}
    overall = _stats(set(classes))
    return {
        'iou_thresh': iou_thresh,
        'overall': overall,
        'per_class': per_class,
        'matches': [{'pred': pi, 'gt': gi, 'iou': round(iou, 3)}
                    for pi, gi, iou in matches],
    }
