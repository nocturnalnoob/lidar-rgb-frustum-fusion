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


def poly_iou(poly_a, poly_b):
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


def gt_bev_corners(obj, calib):
    """Ground-truth KITTI object -> (4,2) BEV corners in the Velodyne frame."""
    corners_rect = compute_box_3d_corners_rect(
        obj['dimensions'], obj['location'], obj['rotation_y'])
    return rect_to_velo(corners_rect, calib)[:4, :2]


# Backwards-compatible aliases (older call sites used the underscore names).
_poly_iou = poly_iou
_gt_bev_corners = gt_bev_corners


def average_precision(tp_flags, n_gt):
    """
    VOC 2010+ style all-point AP from a score-sorted list of TP/FP flags.

    Args:
        tp_flags: list/array of 1 (true positive) / 0 (false positive), ordered
                  by descending prediction confidence across the whole dataset.
        n_gt:     total number of ground-truth objects for this class.
    Returns:
        (ap, recall_curve, precision_curve)
    """
    tp = np.asarray(tp_flags, dtype=np.float64)
    if n_gt == 0:
        return 0.0, np.array([0.0]), np.array([1.0])
    if tp.size == 0:
        return 0.0, np.array([0.0]), np.array([1.0])
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # Envelope (make precision monotonically decreasing), then integrate.
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, recall, precision


def kitti_difficulty(obj):
    """
    Official KITTI difficulty for a labelled object, from 2D box height,
    occlusion and truncation. Returns 0 (Easy), 1 (Moderate), 2 (Hard), or
    None (below Hard -> ignored in evaluation).
    """
    h = obj['bbox'][3] - obj['bbox'][1]
    occ, trunc = obj['occluded'], obj['truncated']
    if h >= 40 and occ <= 0 and trunc <= 0.15:
        return 0
    if h >= 25 and occ <= 1 and trunc <= 0.30:
        return 1
    if h >= 25 and occ <= 2 and trunc <= 0.50:
        return 2
    return None


def bbox_iou_2d(a, b):
    """IoU of two axis-aligned image boxes [x1,y1,x2,y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


# Neighbouring classes that must NOT be counted as false positives (KITTI rule).
_NEIGHBORS = {'Car': {'Van'}, 'Pedestrian': {'Person_sitting'}, 'Cyclist': set()}


def evaluate_kitti(frame_records, classes=('Car', 'Pedestrian', 'Cyclist'),
                   iou_thresh=0.5, difficulty=1):
    """
    KITTI-style BEV AP at a given difficulty level.

    Objects harder than `difficulty`, neighbour classes (Van/Person_sitting),
    and DontCare regions are treated as "ignore": a detection matching them is
    neither a true nor a false positive. This is the standard KITTI protocol,
    applied in the bird's-eye plane.

    Args:
        frame_records: list per frame of {'preds': [...], 'gts': [...]}, where
            each pred has keys kitti_class, score, bbox (2D), bev (4,2), and each
            gt has type, bbox (2D), difficulty, bev (4,2) or None.
    Returns:
        {class: {ap, n_gt, tp, recall, precision, match_iou[], center_err[]}},
        plus 'mAP'.
    """
    result = {}
    aps = []
    for cls in classes:
        n_gt = 0
        preds = []
        per_frame = {}
        for fi, rec in enumerate(frame_records):
            valid, ign_bev, ign_2d = [], [], []
            for o in rec['gts']:
                if o['type'] == cls:
                    d = o['difficulty']
                    if d is not None and d <= difficulty:
                        valid.append({'bev': o['bev'], 'used': False,
                                      'loc': o.get('loc')})
                    elif o['bev'] is not None:
                        ign_bev.append(o['bev'])          # harder -> ignore
                elif o['type'] in _NEIGHBORS[cls] and o['bev'] is not None:
                    ign_bev.append(o['bev'])
                elif o['type'] == 'DontCare':
                    ign_2d.append(o['bbox'])
            n_gt += len(valid)
            per_frame[fi] = {'valid': valid, 'ign_bev': ign_bev, 'ign_2d': ign_2d}
            for p in rec['preds']:
                if p['kitti_class'] == cls:
                    preds.append((p['score'], fi, p['bev'], p['bbox'], p.get('loc')))

        preds.sort(key=lambda x: -x[0])
        tp_flags, match_iou, center_err = [], [], []
        for score, fi, bev, bbox, ploc in preds:
            pf = per_frame[fi]
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(pf['valid']):
                if g['used']:
                    continue
                iou = poly_iou(bev, g['bev'])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= iou_thresh:
                pf['valid'][best_j]['used'] = True
                tp_flags.append(1)
                match_iou.append(best_iou)
                g = pf['valid'][best_j]
                if ploc is not None and g.get('loc') is not None:
                    center_err.append(float(np.hypot(ploc[0] - g['loc'][0],
                                                      ploc[1] - g['loc'][1])))
                continue
            # Ignore (harder GT / neighbour class / DontCare region)?
            ignored = any(poly_iou(bev, gb) >= iou_thresh for gb in pf['ign_bev'])
            if not ignored:
                ignored = any(bbox_iou_2d(bbox, db) >= 0.5 for db in pf['ign_2d'])
            if ignored:
                continue
            tp_flags.append(0)

        ap, recall, precision = average_precision(tp_flags, n_gt)
        result[cls] = {
            'ap': round(ap, 4), 'n_gt': n_gt, 'tp': int(sum(tp_flags)),
            'recall': recall.tolist(), 'precision': precision.tolist(),
            'match_iou': match_iou, 'center_err': center_err,
        }
        if n_gt > 0:
            aps.append(ap)
    result['mAP'] = round(float(np.mean(aps)), 4) if aps else 0.0
    return result


def evaluate_dataset(predictions, ground_truths, classes=('Car', 'Pedestrian', 'Cyclist'),
                     iou_thresh=0.5):
    """
    Compute per-class AP over a whole dataset in the bird's-eye plane.

    Args:
        predictions:   list of dicts {frame, kitti_class, score, bev (4,2)}.
        ground_truths: list of dicts {frame, type, bev (4,2)}.
        iou_thresh:    BEV IoU required for a true positive.
    Returns:
        dict: {class: {ap, n_gt, n_pred, recall, precision}}, plus 'mAP'.
    """
    result = {}
    aps = []
    for cls in classes:
        preds = [p for p in predictions if p['kitti_class'] == cls]
        preds.sort(key=lambda p: -p['score'])
        # Ground truth grouped by frame, with a matched flag.
        gt_by_frame = {}
        n_gt = 0
        for g in ground_truths:
            if g['type'] != cls:
                continue
            gt_by_frame.setdefault(g['frame'], []).append({'bev': g['bev'], 'used': False})
            n_gt += 1

        tp_flags = []
        for p in preds:
            gts = gt_by_frame.get(p['frame'], [])
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if g['used']:
                    continue
                iou = poly_iou(p['bev'], g['bev'])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= iou_thresh:
                gts[best_j]['used'] = True
                tp_flags.append(1)
            else:
                tp_flags.append(0)

        ap, recall, precision = average_precision(tp_flags, n_gt)
        result[cls] = {
            'ap': round(ap, 4),
            'n_gt': n_gt,
            'n_pred': len(preds),
            'recall': recall.tolist(),
            'precision': precision.tolist(),
        }
        if n_gt > 0:
            aps.append(ap)

    result['mAP'] = round(float(np.mean(aps)), 4) if aps else 0.0
    return result


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
