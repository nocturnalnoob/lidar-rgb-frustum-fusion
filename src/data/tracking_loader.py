"""
Loader for a single KITTI *tracking* sequence.

The tracking benchmark differs from the object-detection split in two ways that
are normalized here at the loader boundary so the rest of the code (calibration,
fusion) stays single-path:

  1. Calibration keys are named `R_rect` / `Tr_velo_cam` (vs `R0_rect` /
     `Tr_velo_to_cam` in the detection split). We remap them on read.
  2. Labels live in one file per sequence with two extra leading columns
     (frame index, track id). We parse that layout separately.
"""

import os
import numpy as np
import cv2


class TrackingLoader:
    def __init__(self, base_dir, seq='0000', split='training'):
        self.seq = seq
        root = os.path.join(base_dir, split)
        self.calib_file = os.path.join(root, 'calib', f'{seq}.txt')
        self.image_dir = os.path.join(root, 'image_02', seq)
        self.velodyne_dir = os.path.join(root, 'velodyne', seq)
        self.label_file = os.path.join(root, 'label_02', f'{seq}.txt')
        self._labels_by_frame = None

    def list_frames(self):
        if not os.path.isdir(self.velodyne_dir):
            return []
        ids = [int(os.path.splitext(f)[0]) for f in os.listdir(self.velodyne_dir)
               if f.endswith('.bin')]
        return sorted(ids)

    def get_calib(self, idx=None):
        """Parse sequence calibration, normalized to the detection-split keys."""
        raw = {}
        with open(self.calib_file, 'r') as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                else:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    k, v = parts[0], ' '.join(parts[1:])
                k = k.strip()
                try:
                    raw[k] = np.array([float(x) for x in v.split()])
                except ValueError:
                    continue

        def pick(*names):
            for n in names:
                if n in raw:
                    return raw[n]
            raise KeyError(f"none of {names} in calib {self.calib_file}")

        calib = {
            'P2': pick('P2').reshape(3, 4),
            'R0_rect': pick('R_rect', 'R0_rect').reshape(3, 3),
            'Tr_velo_to_cam': pick('Tr_velo_cam', 'Tr_velo_to_cam').reshape(3, 4),
        }
        return calib

    def get_image(self, idx):
        path = os.path.join(self.image_dir, f'{idx:06d}.png')
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def get_lidar(self, idx):
        path = os.path.join(self.velodyne_dir, f'{idx:06d}.bin')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    def _load_labels(self):
        """Parse the whole-sequence label file into {frame_idx: [objects]}."""
        by_frame = {}
        if not os.path.exists(self.label_file):
            return by_frame
        with open(self.label_file, 'r') as f:
            for line in f:
                p = line.split()
                if len(p) < 17:
                    continue
                frame = int(p[0])
                by_frame.setdefault(frame, []).append({
                    'track_id': int(p[1]),
                    'type': p[2],
                    'truncated': float(p[3]),
                    'occluded': int(float(p[4])),
                    'alpha': float(p[5]),
                    'bbox': np.array([float(x) for x in p[6:10]]),
                    'dimensions': np.array([float(x) for x in p[10:13]]),  # h,w,l
                    'location': np.array([float(x) for x in p[13:16]]),    # x,y,z
                    'rotation_y': float(p[16]),
                })
        return by_frame

    def has_labels(self, idx):
        return os.path.exists(self.label_file)

    def get_labels(self, idx):
        if self._labels_by_frame is None:
            self._labels_by_frame = self._load_labels()
        return self._labels_by_frame.get(idx, [])
