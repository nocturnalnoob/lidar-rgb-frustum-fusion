import numpy as np
import cv2
import matplotlib.pyplot as plt

def project_velo_to_image(pts_3d_velo, calib):
    """
    Projects 3D LiDAR points in Velodyne frame to 2D image plane.
    Formula: Y = P2 * R0_rect * Tr_velo_to_cam * X
    
    Args:
        pts_3d_velo: np.array of shape (N, 3) or (N, 4) (x, y, z, reflectance)
        calib: dict containing P2, R0_rect, Tr_velo_to_cam matrices
    Returns:
        pts_2d: (M, 2) array of projected X, Y pixel coordinates
        pts_cam_rect: (M, 3) 3D points in rectified camera coordinates (for depth)
        mask: boolean mask (M,) for points strictly in front of the camera
    """
    # Tr_velo_to_cam: 3x4 -> 4x4
    Tr = np.eye(4)
    Tr[:3, :4] = calib['Tr_velo_to_cam']
    
    # R0_rect: 3x3 -> 4x4
    R0 = np.eye(4)
    R0[:3, :3] = calib['R0_rect']
    
    # P2: 3x4
    P2 = calib['P2']
    
    # Convert points to homogeneous coordinates: (X, Y, Z, 1)
    N = pts_3d_velo.shape[0]
    pts_3d_hom = np.ones((N, 4))
    pts_3d_hom[:, :3] = pts_3d_velo[:, :3]
    
    # Project to Camera Rectified coordinate system (R0_rect * Tr_velo_to_cam * X)
    pts_cam_hom = (R0 @ Tr @ pts_3d_hom.T).T  # Shape: (N, 4)
    
    # Keep only points in front of the camera (Z > 0)
    # Typically, KITTI cameras are at Z=0, looking down positive Z-axis.
    mask_front = pts_cam_hom[:, 2] > 0.1
    pts_cam_front = pts_cam_hom[mask_front]
    
    # Project to image plane
    pts_img_hom = (P2 @ pts_cam_front.T).T  # Shape: (M, 3)
    
    # Normalize by depth (Z) to get 2D image coordinates (X/Z, Y/Z)
    pts_2d = pts_img_hom[:, :2] / pts_img_hom[:, 2:]
    
    return pts_2d, pts_cam_front[:, :3], mask_front

def velo_to_rect(pts_velo, calib):
    """
    Transform points from the Velodyne frame to the rectified camera frame.

    Args:
        pts_velo: (N, 3) array of [x, y, z] in the Velodyne frame.
        calib: dict with R0_rect (3x3) and Tr_velo_to_cam (3x4).
    Returns:
        (N, 3) array in the rectified camera frame.
    """
    pts_velo = np.asarray(pts_velo, dtype=np.float64).reshape(-1, 3)
    Tr = calib['Tr_velo_to_cam']            # (3, 4)
    R0 = calib['R0_rect']                   # (3, 3)
    pts_hom = np.hstack([pts_velo, np.ones((pts_velo.shape[0], 1))])
    pts_cam = (Tr @ pts_hom.T).T            # (N, 3) in unrectified camera frame
    pts_rect = (R0 @ pts_cam.T).T           # (N, 3) in rectified camera frame
    return pts_rect


def rect_to_velo(pts_rect, calib):
    """
    Inverse of velo_to_rect: rectified camera frame -> Velodyne frame.
    """
    pts_rect = np.asarray(pts_rect, dtype=np.float64).reshape(-1, 3)
    Tr = calib['Tr_velo_to_cam']
    R0 = calib['R0_rect']
    R = Tr[:3, :3]                          # rotation velo->cam
    t = Tr[:3, 3]                           # translation velo->cam
    pts_cam = (R0.T @ pts_rect.T).T         # undo rectification (R0 orthonormal)
    pts_velo = (R.T @ (pts_cam - t).T).T    # undo velo->cam rigid transform
    return pts_velo


def project_rect_to_image(pts_rect, calib):
    """
    Project rectified camera-frame points onto the image plane via P2.

    Returns:
        pts_2d: (N, 2) pixel coordinates.
        depth:  (N,) depth (Z in rect frame) for each point.
    """
    pts_rect = np.asarray(pts_rect, dtype=np.float64).reshape(-1, 3)
    P2 = calib['P2']
    pts_hom = np.hstack([pts_rect, np.ones((pts_rect.shape[0], 1))])
    pts_img = (P2 @ pts_hom.T).T            # (N, 3)
    depth = pts_img[:, 2]
    pts_2d = pts_img[:, :2] / depth[:, None]
    return pts_2d, depth


def project_velo_to_image_indexed(pts_velo, calib):
    """
    Project every Velodyne point to the image WITHOUT dropping any, preserving
    one-to-one correspondence with the input rows. Used for frustum selection.

    Args:
        pts_velo: (N, 3+) array; only the first 3 columns are used.
    Returns:
        pts_2d:    (N, 2) pixel coordinates (meaningless where in_front is False).
        depth:     (N,) depth in the rectified camera frame.
        in_front:  (N,) bool mask, True where the point is in front of the camera.
    """
    pts_rect = velo_to_rect(pts_velo[:, :3], calib)
    in_front = pts_rect[:, 2] > 0.1
    P2 = calib['P2']
    pts_hom = np.hstack([pts_rect, np.ones((pts_rect.shape[0], 1))])
    pts_img = (P2 @ pts_hom.T).T
    depth = pts_img[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    pts_2d = pts_img[:, :2] / safe[:, None]
    return pts_2d, pts_rect[:, 2], in_front


def roty(angle):
    """3x3 rotation matrix about the camera Y (down) axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def compute_box_3d_corners_rect(dims_hwl, location, ry):
    """
    Compute the 8 corners of a KITTI 3D box in the rectified camera frame.

    Args:
        dims_hwl: (h, w, l) in metres.
        location: (x, y, z) bottom-centre of the box in rect coords.
        ry:       rotation about the camera Y axis (radians).
    Returns:
        (8, 3) corner coordinates in the rectified camera frame.
    """
    h, w, l = dims_hwl
    x_c = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_c = [0, 0, 0, 0, -h, -h, -h, -h]
    z_c = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners = roty(ry) @ np.vstack([x_c, y_c, z_c])
    corners = corners.T + np.asarray(location, dtype=np.float64)
    return corners


def overlay_lidar_on_image(image, pts_2d, depth, max_depth=70.0):
    """
    Overlays projected LiDAR points onto an RGB image, colored by their depth.
    """
    img_out = image.copy()
    h, w = img_out.shape[:2]
    
    # Normalize depth for colormap (clip between 0 and max_depth)
    depth_norm = np.clip(depth / max_depth, 0, 1)
    
    # Grab standard colormap from matplotlib
    cmap = plt.get_cmap('jet')
    # Apply colormap (RGBA) and convert to BGR for OpenCV
    colors = (cmap(1 - depth_norm)[:, :3] * 255)[:, ::-1]  # BGR order
    
    for i in range(len(pts_2d)):
        u, v = int(np.round(pts_2d[i, 0])), int(np.round(pts_2d[i, 1]))
        
        # Draw point if within image boundaries
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img_out, (u, v), 1, color=colors[i].tolist(), thickness=-1)
            
    return img_out
