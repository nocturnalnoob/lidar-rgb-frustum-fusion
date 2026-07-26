"""
Late-fusion 3D object detection.

Pipeline per frame:
  1. YOLOv8 produces 2D boxes on the RGB image (RGB modality).
  2. Every LiDAR point is projected into the image (LiDAR modality + calibration).
  3. For each 2D box we take the LiDAR points whose projection lands inside it
     (a viewing "frustum"), strip the ground plane, and cluster the remainder.
  4. The dominant cluster -> an oriented 3D bounding box in the camera frame.

This turns a 2D detector into a 3D detector by *fusing* the camera's strong
semantics with the LiDAR's accurate geometry — without training a 3D network.
"""

import numpy as np
from sklearn.cluster import DBSCAN

from src.calibration.project_lidar import (
    project_velo_to_image_indexed,
    velo_to_rect,
    rect_to_velo,
)

# KITTI class-median dimensions (h, w, l) in metres. A single LiDAR sweep only
# observes an object's near faces, so we refine the measured extent toward these
# priors (position and heading still come from the LiDAR geometry).
CLASS_PRIORS = {
    'Car': (1.53, 1.63, 3.88),
    'Pedestrian': (1.76, 0.66, 0.84),
    'Cyclist': (1.74, 0.60, 1.76),
}


def segment_ground(pts_velo, dist_thresh=0.2, n_iters=120, seed_height=-1.2, seed=0):
    """
    Estimate the ground plane in the Velodyne frame with RANSAC.

    The Velodyne frame has +z up, so the ground is a near-horizontal plane a bit
    below the sensor.

    Returns:
        non_ground: (N,) bool mask, True for points NOT on the ground.
        plane:      (normal(3,), d) of the fitted ground plane, or None.
    """
    pts = pts_velo[:, :3]
    n = pts.shape[0]
    if n < 50:
        return np.ones(n, dtype=bool), None

    # Seed RANSAC from low points only — robust against walls/large objects.
    candidate_idx = np.where(pts[:, 2] < seed_height + 1.0)[0]
    if candidate_idx.size < 50:
        candidate_idx = np.arange(n)

    rng = np.random.default_rng(seed)
    best_inliers, best_count, best_plane = None, 0, None
    for _ in range(n_iters):
        a, b, c = candidate_idx[rng.integers(0, candidate_idx.size, size=3)]
        p1, p2, p3 = pts[a], pts[b], pts[c]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        # Require a roughly horizontal plane (normal close to +/- z).
        if abs(normal[2]) < 0.85:
            continue
        d = -normal.dot(p1)
        dist = np.abs(pts @ normal + d)
        inliers = dist < dist_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
            best_plane = (normal, float(d))

    if best_inliers is None:
        return pts[:, 2] > seed_height, None
    return ~best_inliers, best_plane


def _ground_z(plane, x, y):
    """Height (z) of the ground plane at Velodyne coordinate (x, y)."""
    if plane is None:
        return None
    (nx, ny, nz), d = plane
    if abs(nz) < 1e-6:
        return None
    return -(nx * x + ny * y + d) / nz


def _frustum_mask(pts_2d, in_front, depth, bbox, shrink=0.05, max_depth=70.0):
    """Boolean mask of points projecting inside a (slightly shrunk) 2D box."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x1s, x2s = x1 + shrink * w, x2 - shrink * w
    y1s, y2s = y1 + shrink * h, y2 - shrink * h
    u, v = pts_2d[:, 0], pts_2d[:, 1]
    return (
        in_front
        & (depth < max_depth)
        & (u >= x1s) & (u <= x2s)
        & (v >= y1s) & (v <= y2s)
    )


def _dominant_cluster(pts_velo, eps=0.6, min_samples=8):
    """
    DBSCAN the frustum points and pick the object cluster: the closest cluster
    (smallest median range) that is reasonably sized. Returns indices into
    pts_velo, or None.
    """
    if pts_velo.shape[0] < min_samples:
        return None
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_velo[:, :3])
    valid = labels >= 0
    if not valid.any():
        return None

    best_idx = None
    best_range = np.inf
    for lab in np.unique(labels[valid]):
        idx = np.where(labels == lab)[0]
        if idx.size < min_samples:
            continue
        rng = np.median(np.linalg.norm(pts_velo[idx, :2], axis=1))
        if rng < best_range:
            best_range = rng
            best_idx = idx
    return best_idx


def _oriented_box_from_cluster(cluster_velo, kitti_class=None, ground_plane=None):
    """
    Fit an oriented 3D box to a Velodyne cluster.

    Heading comes from the minimum-area rectangle in the bird's-eye (x-y) plane;
    position from the cluster. Because a LiDAR sweep only sees an object's near
    faces, the measured extent is refined toward class-size priors and the box is
    re-anchored so it grows *away* from the sensor (keeping the observed near face
    fixed). The box floor is snapped to the estimated ground plane.

    Returns a dict in the Velodyne frame:
        center_bottom (3,), size (l, w, h), yaw, corners (8,3).
    """
    import cv2

    xy = cluster_velo[:, :2].astype(np.float32)
    z = cluster_velo[:, 2]
    rect = cv2.minAreaRect(xy)            # ((cx,cy),(d1,d2),angle_deg)
    (cx, cy), _, _ = rect
    box_pts = cv2.boxPoints(rect)

    e1 = box_pts[1] - box_pts[0]
    e2 = box_pts[2] - box_pts[1]
    if np.linalg.norm(e1) >= np.linalg.norm(e2):
        long_edge, length, width = e1, np.linalg.norm(e1), np.linalg.norm(e2)
    else:
        long_edge, length, width = e2, np.linalg.norm(e2), np.linalg.norm(e1)
    yaw = float(np.arctan2(long_edge[1], long_edge[0]))

    # Height: floor on the ground plane, top from the observed points.
    prior = CLASS_PRIORS.get(kitti_class)
    gz = _ground_z(ground_plane, cx, cy)
    z_bottom = gz if gz is not None else float(z.min())
    z_top = float(z.max())
    height = z_top - z_bottom
    if prior is not None:
        length = max(length, prior[2])
        width = max(width, prior[1])
        height = max(height, prior[0])
    z_top = z_bottom + height

    center = np.array([cx, cy], dtype=np.float64)
    u = np.array([np.cos(yaw), np.sin(yaw)])          # heading axis
    # Re-anchor along heading: keep the near end fixed, grow the far end out.
    half0 = np.linalg.norm(long_edge) / 2.0
    end_a, end_b = center + half0 * u, center - half0 * u
    if np.linalg.norm(end_a) <= np.linalg.norm(end_b):
        near, far_dir = end_a, -u
    else:
        near, far_dir = end_b, u
    center = near + far_dir * (length / 2.0)

    # Build the 4 BEV corners from (center, yaw, length, width).
    v = np.array([-np.sin(yaw), np.cos(yaw)])          # width axis
    hl, hw = length / 2.0, width / 2.0
    bev = np.array([center + hl * u + hw * v,
                    center + hl * u - hw * v,
                    center - hl * u - hw * v,
                    center - hl * u + hw * v])
    bottom = np.column_stack([bev, np.full(4, z_bottom)])
    top = np.column_stack([bev, np.full(4, z_top)])
    corners = np.vstack([bottom, top])

    return {
        'center_bottom': np.array([center[0], center[1], z_bottom]),
        'size': (float(length), float(width), float(height)),  # l, w, h
        'yaw': yaw,
        'corners_velo': corners,
    }


def _to_kitti_box(box_velo, calib):
    """Convert a Velodyne-frame box into KITTI rect-frame label fields."""
    loc_rect = velo_to_rect(box_velo['center_bottom'], calib)[0]
    l, w, h = box_velo['size']
    # KITTI rotation_y relates to the Velodyne heading yaw by ry = -yaw - pi/2.
    ry = -box_velo['yaw'] - np.pi / 2
    ry = (ry + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    return {
        'location': loc_rect,                 # bottom-centre in rect frame
        'dimensions': np.array([h, w, l]),    # KITTI order: h, w, l
        'rotation_y': float(ry),
    }


def fuse_frame(lidar, calib, detections,
               ground_thresh=0.2, dbscan_eps=0.6, min_cluster=8,
               max_depth=70.0):
    """
    Run frustum late-fusion for all 2D detections in a frame.

    Args:
        lidar:      (N, 4) raw Velodyne scan.
        calib:      calibration dict from KittiLoader.get_calib.
        detections: list from YoloDetector.detect.
    Returns:
        List of 3D detections, each augmented with:
            corners_velo (8,3), location, dimensions (h,w,l), rotation_y,
            depth, n_points, plus the original 2D fields.
    """
    pts_2d, depth, in_front = project_velo_to_image_indexed(lidar, calib)

    # Global ground segmentation once per frame (reused by every frustum).
    non_ground, ground_plane = segment_ground(lidar, dist_thresh=ground_thresh)

    results = []
    for det in detections:
        fmask = _frustum_mask(pts_2d, in_front, depth, det['bbox'],
                              max_depth=max_depth)
        obj_mask = fmask & non_ground
        cluster_pts = lidar[obj_mask][:, :3]

        cluster_idx = _dominant_cluster(cluster_pts, eps=dbscan_eps,
                                        min_samples=min_cluster)
        if cluster_idx is None:
            continue
        cluster = cluster_pts[cluster_idx]

        box_velo = _oriented_box_from_cluster(
            cluster, kitti_class=det['kitti_class'], ground_plane=ground_plane)
        kitti = _to_kitti_box(box_velo, calib)
        obj_depth = float(np.median(np.linalg.norm(cluster[:, :2], axis=1)))

        out = dict(det)
        out.update({
            'corners_velo': box_velo['corners_velo'],
            'center_bottom_velo': box_velo['center_bottom'],
            'size_lwh': box_velo['size'],
            'yaw_velo': box_velo['yaw'],
            'location': kitti['location'],
            'dimensions': kitti['dimensions'],
            'rotation_y': kitti['rotation_y'],
            'depth': obj_depth,
            'n_points': int(cluster.shape[0]),
        })
        results.append(out)
    return results
