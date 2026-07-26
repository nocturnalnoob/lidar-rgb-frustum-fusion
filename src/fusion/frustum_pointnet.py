"""
A compact Frustum-PointNet-style 3D box head.

Given the LiDAR points inside a 2D detection's frustum, a small PointNet regresses
the object's 3D box. Following Frustum-PointNet (Qi et al., 2018) we normalize the
viewpoint by rotating the frustum to a canonical heading before the network sees
it, and regress residuals against a class-mean size anchor.

This is the *learned* alternative to the geometric box fitter in
`frustum_fusion.py`. It is trained by `scripts/train_frustum.py`.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:  # keeps the package importable without torch
    _HAS_TORCH = False

# Class-mean size anchors (h, w, l) — residuals are regressed against these.
SIZE_ANCHOR = {
    'Car': np.array([1.53, 1.63, 3.88]),
    'Pedestrian': np.array([1.76, 0.66, 0.84]),
    'Cyclist': np.array([1.74, 0.60, 1.76]),
}
CLASS_IDS = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
NUM_POINTS = 256


def frustum_angle(bbox2d, calib):
    """Central viewing angle (about the Velodyne +z axis) of a 2D box.

    Uses the box's horizontal center back-projected through P2; the returned
    angle is used to rotate the frustum to a canonical forward-facing pose.
    """
    P2 = calib['P2']
    u = 0.5 * (bbox2d[0] + bbox2d[2])
    # Ray direction in the rectified camera frame (z forward, x right).
    x = (u - P2[0, 2]) / P2[0, 0]
    ang_cam = np.arctan2(x, 1.0)            # angle from camera forward axis
    # Velodyne x is camera z (forward); rotating frustum by this angle about z.
    return float(ang_cam)


def _rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def normalize_frustum(points_velo, bbox2d, calib):
    """
    Rotate points to a canonical frustum heading and center them.

    Returns:
        pts_norm: (N,3) rotated + mean-subtracted points.
        rot:      the applied rotation angle (about z).
        mean:     the subtracted centroid (in the rotated frame).
    """
    theta = frustum_angle(bbox2d, calib)
    R = _rotz(-theta)
    pts = (R @ points_velo[:, :3].T).T
    mean = pts.mean(axis=0)
    return pts - mean, theta, mean


def sample_points(points, n=NUM_POINTS, rng=None):
    """Fixed-size point set via random sampling (with replacement if sparse)."""
    m = points.shape[0]
    if m == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if rng is None:
        idx = np.random.choice(m, n, replace=(m < n))
    else:
        idx = rng.choice(m, n, replace=(m < n))
    return points[idx].astype(np.float32)


if _HAS_TORCH:

    class FrustumPointNet(nn.Module):
        """PointNet: per-point MLP -> global max-pool -> box head."""

        def __init__(self, num_classes=3):
            super().__init__()
            self.feat = nn.Sequential(
                nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            )
            # + one-hot class vector appended to the global feature.
            self.head = nn.Sequential(
                nn.Linear(256 + num_classes, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 3 + 3 + 2),   # center(3) + size resid(3) + (sin,cos)
            )

        def forward(self, pts, cls_onehot):
            # pts: (B, N, 3) -> (B, 3, N)
            x = self.feat(pts.transpose(1, 2))
            x = torch.max(x, dim=2).values            # (B, 256)
            x = torch.cat([x, cls_onehot], dim=1)
            out = self.head(x)
            center = out[:, :3]
            size_resid = out[:, 3:6]
            heading = out[:, 6:8]                      # (sin, cos), un-normalized
            return center, size_resid, heading

    def decode_prediction(center, size_resid, heading, kitti_class, rot, mean):
        """Network output (canonical frame) -> box in the Velodyne frame."""
        anchor = SIZE_ANCHOR[kitti_class]
        h, w, l = (anchor + size_resid)
        yaw_canon = float(np.arctan2(heading[0], heading[1]))
        # Undo the frustum rotation applied in normalize_frustum.
        center_rot = center + mean
        R = _rotz(rot)
        center_velo = R @ center_rot
        yaw_velo = yaw_canon + rot
        return {'center_velo': center_velo, 'size_hwl': np.array([h, w, l]),
                'yaw_velo': yaw_velo}

    def load_frustum_model(weights, num_classes=3):
        """Load a trained FrustumPointNet in eval mode (CPU)."""
        model = FrustumPointNet(num_classes=num_classes)
        model.load_state_dict(torch.load(weights, map_location='cpu'))
        model.eval()
        return model

    def predict_box(model, points_velo, bbox2d, calib, kitti_class):
        """Run the learned head on a frustum's raw points -> box in the Velodyne frame.

        Returns {center_velo (box centre), size_hwl, yaw_velo}. The input points are
        the raw 2D-box crop (same distribution the model was trained on): no ground
        removal or clustering — the PointNet handles that implicitly.
        """
        pts_norm, theta, mean = normalize_frustum(points_velo, bbox2d, calib)
        pts = sample_points(pts_norm, NUM_POINTS)
        onehot = np.zeros(3, dtype=np.float32)
        onehot[CLASS_IDS[kitti_class]] = 1
        with torch.no_grad():
            c, s, h = model(torch.tensor(pts)[None], torch.tensor(onehot)[None])
            h = (h / h.norm(dim=1, keepdim=True).clamp_min(1e-6))[0].numpy()
        return decode_prediction(c[0].numpy(), s[0].numpy(), h, kitti_class, theta, mean)
