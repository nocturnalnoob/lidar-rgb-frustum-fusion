"""
Train the compact Frustum-PointNet 3D box head on the downloaded KITTI frames,
then compare it against the geometric fitter TERM BY TERM (center / heading /
size) on a held-out, frame-level split.

Both methods consume the same GT-2D-box frustums, so the comparison isolates the
box estimator. Because the geometric fitter takes size from class priors and the
learned head regresses size residuals against the same anchor, size is expected
to roughly tie; heading is where a learned model can plausibly beat the
min-area-rectangle fit.

Usage:
    python scripts/train_frustum.py --data_dir data/kitti --epochs 120
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn

from src.data.kitti_loader import KittiLoader
from src.calibration.project_lidar import project_velo_to_image_indexed, rect_to_velo
from src.fusion.frustum_fusion import fuse_frame
from src.fusion.frustum_pointnet import (
    FrustumPointNet, SIZE_ANCHOR, CLASS_IDS, NUM_POINTS,
    normalize_frustum, sample_points, decode_prediction, _rotz)

CLASSES = ('Car', 'Pedestrian', 'Cyclist')
MIN_PTS = 40


def build_samples(loader, frames):
    """Extract frustum samples (using GT 2D boxes) with their GT 3D box."""
    samples = []
    for idx in frames:
        calib = loader.get_calib(idx)
        lidar = loader.get_lidar(idx)
        pts2d, depth, front = project_velo_to_image_indexed(lidar, calib)
        for o in loader.get_labels(idx):
            if o['type'] not in CLASSES or o['dimensions'][0] <= 0:
                continue
            x1, y1, x2, y2 = o['bbox']
            m = (front & (pts2d[:, 0] >= x1) & (pts2d[:, 0] <= x2)
                 & (pts2d[:, 1] >= y1) & (pts2d[:, 1] <= y2) & (depth < 70))
            pts = lidar[m][:, :3]
            if pts.shape[0] < MIN_PTS:
                continue
            bottom = rect_to_velo(o['location'], calib)[0]
            h = o['dimensions'][0]
            center_velo = bottom + np.array([0, 0, h / 2.0])   # box centre in velo
            yaw_velo = -o['rotation_y'] - np.pi / 2
            samples.append({'frame': idx, 'calib': calib, 'points': pts,
                            'bbox': o['bbox'], 'cls': o['type'],
                            'center': center_velo, 'size': o['dimensions'].copy(),
                            'yaw': yaw_velo})
    return samples


def encode(sample, rng):
    """Sample -> (points, class one-hot, targets) in the canonical frustum frame."""
    pts_norm, theta, mean = normalize_frustum(sample['points'], sample['bbox'], sample['calib'])
    pts = sample_points(pts_norm, NUM_POINTS, rng)
    center_rot = _rotz(-theta) @ sample['center'] - mean
    yaw_canon = sample['yaw'] - theta
    onehot = np.zeros(3, dtype=np.float32)
    onehot[CLASS_IDS[sample['cls']]] = 1
    return (pts, onehot, center_rot.astype(np.float32),
            (sample['size'] - SIZE_ANCHOR[sample['cls']]).astype(np.float32),
            np.array([np.sin(yaw_canon), np.cos(yaw_canon)], dtype=np.float32))


def angle_err(a, b):
    d = abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return min(d, abs(np.pi - d))   # box is symmetric under a pi rotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/kitti')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--out', default='models/frustum_pointnet.pt')
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    loader = KittiLoader(args.data_dir)
    frames = loader.list_frames()
    val_frames = [f for f in frames if f % 5 == 0]
    train_frames = [f for f in frames if f % 5 != 0]

    print(f"Extracting frustums (>= {MIN_PTS} pts) ...")
    train = build_samples(loader, train_frames)
    val = build_samples(loader, val_frames)
    print(f"  train={len(train)}  val={len(val)}")
    for c in CLASSES:
        print(f"    {c}: train={sum(s['cls']==c for s in train)} "
              f"val={sum(s['cls']==c for s in val)}")

    P, O, TC, TS, TH = [], [], [], [], []
    for s in train:
        p, o, tc, ts, th = encode(s, rng)
        P.append(p); O.append(o); TC.append(tc); TS.append(ts); TH.append(th)
    P, O = torch.tensor(np.stack(P)), torch.tensor(np.stack(O))
    TC, TS, TH = torch.tensor(np.stack(TC)), torch.tensor(np.stack(TS)), torch.tensor(np.stack(TH))

    model = FrustumPointNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sl1 = nn.SmoothL1Loss()
    n, bs = P.shape[0], 32
    print("Training ...")
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            pc, ps, ph = model(P[b], O[b])
            ph = ph / ph.norm(dim=1, keepdim=True).clamp_min(1e-6)
            loss = sl1(pc, TC[b]) + sl1(ps, TS[b]) + sl1(ph, TH[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{args.epochs} loss={tot/n:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved model -> {args.out}")

    # ---- Term-by-term evaluation on val: learned vs geometric ----
    model.eval()
    learned = {'center': [], 'yaw': [], 'size': []}
    geom = {'center': [], 'yaw': [], 'size': []}
    with torch.no_grad():
        for s in val:
            p, o, _, _, _ = encode(s, rng)
            pc, ps, ph = model(torch.tensor(p)[None], torch.tensor(o)[None])
            ph = (ph / ph.norm(dim=1, keepdim=True).clamp_min(1e-6))[0].numpy()
            _, theta, mean = normalize_frustum(s['points'], s['bbox'], s['calib'])
            box = decode_prediction(pc[0].numpy(), ps[0].numpy(), ph, s['cls'], theta, mean)
            learned['center'].append(np.hypot(*(box['center_velo'][:2] - s['center'][:2])))
            learned['yaw'].append(angle_err(box['yaw_velo'], s['yaw']))
            learned['size'].append(float(np.mean(box['size_hwl'] / s['size'])))

            det = [{'bbox': s['bbox'], 'kitti_class': s['cls'], 'score': 1.0}]
            g = fuse_frame(loader.get_lidar(s['frame']), s['calib'], det)
            if g:
                gb = g[0]
                gc = gb['corners_velo'][:4, :2].mean(axis=0)
                l, w, h = gb['size_lwh']                       # -> h,w,l order
                geom['center'].append(np.hypot(*(gc - s['center'][:2])))
                geom['yaw'].append(angle_err(gb['yaw_velo'], s['yaw']))
                geom['size'].append(float(np.mean(np.array([h, w, l]) / s['size'])))

    med = lambda a: round(float(np.median(a)), 3) if a else None
    report = {
        'val_n': len(val),
        'learned_median': {k: med(v) for k, v in learned.items()},
        'geometric_median': {k: med(v) for k, v in geom.items()},
        'per_class_val': {c: sum(s['cls'] == c for s in val) for c in CLASSES},
        'notes': 'center=BEV centre err (m); yaw=heading err (rad, pi-symmetric); '
                 'size=mean predicted/GT dimension ratio (1.0 ideal).',
    }
    print("\n=== Term-by-term (median), learned vs geometric ===")
    print(json.dumps(report, indent=2))
    os.makedirs('docs', exist_ok=True)
    with open('docs/frustum_eval.json', 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
