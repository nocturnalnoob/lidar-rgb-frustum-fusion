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


def segment_ground(pts_velo, dist_thresh=0.2, n_iters=80, seed_height=-1.2):
    """
    Estimate the ground plane in the Velodyne frame with RANSAC.

    The Velodyne frame has +z up, so the ground is a near-horizontal plane a bit
    below the sensor. Returns a boolean mask that is True for NON-ground points.
    """
    pts = pts_velo[:, :3]
    n = pts.shape[0]
    if n < 50:
        return np.ones(n, dtype=bool)

    # Seed RANSAC from low points only — robust against walls/large objects.
    candidate_idx = np.where(pts[:, 2] < seed_height + 1.0)[0]
    if candidate_idx.size < 50:
        candidate_idx = np.arange(n)

    best_inliers = None
    best_count = 0
    rng_order = candidate_idx
    for it in range(n_iters):
        # Deterministic-ish sampling: stride through candidates by iteration.
        a = rng_order[(it * 3) % rng_order.size]
        b = rng_order[(it * 7 + 1) % rng_order.size]
        c = rng_order[(it * 13 + 2) % rng_order.size]
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
            best_count = count
            best_inliers = inliers

    if best_inliers is None:
        # Fallback: simple height threshold.
        return pts[:, 2] > seed_height
    return ~best_inliers


def _frustum_mask(pts_2d, in_front, depth, bbox, shrink=0.10, max_depth=70.0):
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


def _oriented_box_from_cluster(cluster_velo):
    """
    Fit an oriented 3D box to a Velodyne cluster.

    Heading/extent come from the minimum-area rectangle in the bird's-eye (x-y)
    plane; height from the z extent. Returns a dict in the Velodyne frame:
        center_bottom (3,), size (l, w, h), yaw, corners (8,3).
    """
    import cv2

    xy = cluster_velo[:, :2].astype(np.float32)
    z = cluster_velo[:, 2]
    rect = cv2.minAreaRect(xy)            # ((cx,cy),(d1,d2),angle_deg)
    (cx, cy), (d1, d2), _ = rect
    box_pts = cv2.boxPoints(rect)         # (4,2) BEV corners, ordered

    # Heading from the longer edge of the rectangle.
    e1 = box_pts[1] - box_pts[0]
    e2 = box_pts[2] - box_pts[1]
    if np.linalg.norm(e1) >= np.linalg.norm(e2):
        long_edge, length, width = e1, np.linalg.norm(e1), np.linalg.norm(e2)
    else:
        long_edge, length, width = e2, np.linalg.norm(e2), np.linalg.norm(e1)
    yaw = np.arctan2(long_edge[1], long_edge[0])

    z_min, z_max = float(z.min()), float(z.max())
    height = z_max - z_min

    # 8 corners in the Velodyne frame: bottom 4 then top 4.
    bottom = np.column_stack([box_pts, np.full(4, z_min)])
    top = np.column_stack([box_pts, np.full(4, z_max)])
    corners = np.vstack([bottom, top])

    return {
        'center_bottom': np.array([cx, cy, z_min]),
        'size': (float(length), float(width), float(height)),  # l, w, h
        'yaw': float(yaw),
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
    non_ground = segment_ground(lidar, dist_thresh=ground_thresh)

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

        box_velo = _oriented_box_from_cluster(cluster)
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
