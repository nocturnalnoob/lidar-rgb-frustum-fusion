import numpy as np
import cv2
import os

class KittiLoader:
    """
    DataLoader for KITTI 3D Object Detection dataset.
    """
    def __init__(self, base_dir, split='training'):
        self.base_dir = base_dir
        self.split = split
        self.calib_dir = os.path.join(base_dir, split, 'calib')
        self.image_dir = os.path.join(base_dir, split, 'image_2')
        self.velodyne_dir = os.path.join(base_dir, split, 'velodyne')
        self.label_dir = os.path.join(base_dir, split, 'label_2')

    def list_frames(self):
        """Return a sorted list of integer frame indices available on disk."""
        if not os.path.isdir(self.velodyne_dir):
            return []
        ids = []
        for fn in os.listdir(self.velodyne_dir):
            stem, ext = os.path.splitext(fn)
            if ext == '.bin' and stem.isdigit():
                ids.append(int(stem))
        return sorted(ids)

    def has_labels(self, idx):
        return os.path.exists(os.path.join(self.label_dir, f"{idx:06d}.txt"))

    def get_labels(self, idx):
        """
        Parse KITTI ground-truth labels for a frame.

        Returns a list of dicts with keys: type, truncated, occluded, alpha,
        bbox (x1,y1,x2,y2), dimensions (h,w,l), location (x,y,z), rotation_y.
        'DontCare' entries are kept so callers can ignore those image regions.
        """
        label_file = os.path.join(self.label_dir, f"{idx:06d}.txt")
        objects = []
        if not os.path.exists(label_file):
            return objects
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.split()
                if len(parts) < 15:
                    continue
                objects.append({
                    'type': parts[0],
                    'truncated': float(parts[1]),
                    'occluded': int(float(parts[2])),
                    'alpha': float(parts[3]),
                    'bbox': np.array([float(x) for x in parts[4:8]]),
                    'dimensions': np.array([float(x) for x in parts[8:11]]),  # h, w, l
                    'location': np.array([float(x) for x in parts[11:14]]),   # x, y, z (rect)
                    'rotation_y': float(parts[14]),
                })
        return objects

    def get_calib(self, idx):
        """
        Reads calibration matrices from .txt file.
        Returns them as a dictionary of numpy arrays.
        """
        calib_file = os.path.join(self.calib_dir, f"{idx:06d}.txt")
        calib = {}
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                if not line.strip():
                    continue
                key, val = line.split(':', 1)
                calib[key] = np.array([float(x) for x in val.split()])

        # KITTI calibration matrices shape correction
        calib['P2'] = calib['P2'].reshape(3, 4)
        calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)
        calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
        
        return calib

    def get_image(self, idx):
        """
        Reads RGB image.
        """
        img_file = os.path.join(self.image_dir, f"{idx:06d}.png")
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Image not found: {img_file}")
        return cv2.imread(img_file)

    def get_lidar(self, idx):
        """
        Reads point cloud data from .bin file.
        Returns shape (N, 4) with [X, Y, Z, Reflectance]
        """
        lidar_file = os.path.join(self.velodyne_dir, f"{idx:06d}.bin")
        if not os.path.exists(lidar_file):
            raise FileNotFoundError(f"Lidar data not found: {lidar_file}")
        
        # KITTI lidar data is stored as float32
        scan = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        return scan
