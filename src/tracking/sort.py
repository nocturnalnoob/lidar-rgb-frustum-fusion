"""
SORT-style multi-object tracker operating in the bird's-eye plane.

Each track runs a constant-velocity Kalman filter on its BEV center (x, y);
detections are associated to tracks per class by BEV-IoU with Hungarian
matching. Tracks carry a stable integer ID for the lifetime of the object,
which is what makes the tracked visualization legible.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.fusion.metrics import poly_iou


class _KalmanCV:
    """Constant-velocity Kalman filter on 2D position: state [x, y, vx, vy]."""

    def __init__(self, xy):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([1.0, 1.0, 10.0, 10.0])
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                           [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.diag([0.05, 0.05, 0.2, 0.2])
        self.R = np.diag([0.3, 0.3])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, xy):
        z = np.asarray(xy, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


class Track:
    _next_id = 1

    def __init__(self, det):
        self.id = Track._next_id
        Track._next_id += 1
        self.kitti_class = det['kitti_class']
        self.det = det
        self.center = det['corners_velo'][:4, :2].mean(axis=0)
        self.kf = _KalmanCV(self.center)
        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        self.history = [self.center.copy()]     # BEV trail

    def predict(self):
        pred = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return pred

    def update(self, det):
        new_center = det['corners_velo'][:4, :2].mean(axis=0)
        self.kf.update(new_center)
        # Shift the detection box to the smoothed center for display.
        smoothed = self.kf.x[:2]
        shift = smoothed - new_center
        det = dict(det)
        det['corners_velo'] = det['corners_velo'] + np.array([shift[0], shift[1], 0.0])
        self.det = det
        self.center = smoothed.copy()
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.center.copy())

    def predicted_box(self):
        """Last box translated to the Kalman-predicted center (for association)."""
        pred = self.kf.x[:2]
        shift = pred - self.det['corners_velo'][:4, :2].mean(axis=0)
        return self.det['corners_velo'][:4, :2] + shift


class Sort:
    def __init__(self, iou_thresh=0.1, max_age=5, min_hits=2):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []

    def update(self, detections):
        """Advance one frame. Returns the list of active (confirmed) Tracks."""
        for t in self.tracks:
            t.predict()

        # Associate per class with Hungarian matching on BEV IoU.
        unmatched_dets = set(range(len(detections)))
        matched_pairs = []
        for cls in {d['kitti_class'] for d in detections} | {t.kitti_class for t in self.tracks}:
            t_idx = [i for i, t in enumerate(self.tracks) if t.kitti_class == cls]
            d_idx = [j for j, d in enumerate(detections) if d['kitti_class'] == cls]
            if not t_idx or not d_idx:
                continue
            iou = np.zeros((len(t_idx), len(d_idx)))
            for a, ti in enumerate(t_idx):
                pbox = self.tracks[ti].predicted_box()
                for b, dj in enumerate(d_idx):
                    iou[a, b] = poly_iou(pbox, detections[dj]['corners_velo'][:4, :2])
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_thresh:
                    matched_pairs.append((t_idx[r], d_idx[c]))
                    unmatched_dets.discard(d_idx[c])

        for ti, dj in matched_pairs:
            self.tracks[ti].update(detections[dj])
        for dj in unmatched_dets:
            self.tracks.append(Track(detections[dj]))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks
                if t.time_since_update == 0 and t.hits >= self.min_hits]

    @staticmethod
    def reset_ids():
        Track._next_id = 1
