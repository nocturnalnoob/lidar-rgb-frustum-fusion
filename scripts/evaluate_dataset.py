"""
KITTI-style aggregate evaluation of the fusion pipeline.

Runs detection + fusion on every available frame, then reports bird's-eye-view
Average Precision per class and per KITTI difficulty (Easy/Moderate/Hard) using
the standard protocol (harder-than-target GT, neighbour classes, and DontCare
regions are ignored). Also reports localization quality on matched detections
and saves precision-recall curves + a Markdown table for the README.

Usage:
    python scripts/evaluate_dataset.py --data_dir data/kitti --out docs
"""

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.pipeline import FusionPipeline
from src.fusion.metrics import evaluate_kitti, kitti_difficulty, gt_bev_corners

CLASSES = ('Car', 'Pedestrian', 'Cyclist')
# KITTI-standard BEV/3D IoU thresholds.
STD_IOU = {'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}
DIFF_NAMES = ['Easy', 'Moderate', 'Hard']


def build_records(pipe, frames):
    records = []
    t0 = time.time()
    for i, idx in enumerate(frames, 1):
        res = pipe.run(idx, render=False)
        calib = pipe.loader.get_calib(idx)
        preds = []
        for d in res['detections_3d']:
            bev = d['corners_velo'][:4, :2]
            preds.append({'kitti_class': d['kitti_class'], 'score': float(d['score']),
                          'bbox': d['bbox'], 'bev': bev, 'loc': bev.mean(axis=0)})
        gts = []
        for o in res['gt_objects']:
            if o['type'] == 'DontCare':
                gts.append({'type': 'DontCare', 'bbox': o['bbox'],
                            'difficulty': None, 'bev': None, 'loc': None})
                continue
            has_3d = o['dimensions'][0] > 0
            bev = gt_bev_corners(o, calib) if has_3d else None
            gts.append({'type': o['type'], 'bbox': o['bbox'],
                        'difficulty': kitti_difficulty(o), 'bev': bev,
                        'loc': bev.mean(axis=0) if bev is not None else None})
        records.append({'preds': preds, 'gts': gts})
        if i % 20 == 0 or i == len(frames):
            print(f"  processed {i}/{len(frames)} ({i/(time.time()-t0):.1f} fps)")
    return records


def ap_matrix(records):
    """AP[class][difficulty] at each class's standard IoU threshold."""
    out = {}
    curves = {}
    for diff, name in enumerate(DIFF_NAMES):
        # evaluate_kitti runs per its own iou_thresh, so call per class.
        for cls in CLASSES:
            r = evaluate_kitti(records, classes=(cls,), iou_thresh=STD_IOU[cls],
                               difficulty=diff)[cls]
            out.setdefault(cls, {})[name] = r
            if name == 'Moderate':
                curves[cls] = r
    return out, curves


def localization_summary(records):
    """Median IoU / center error / recall on matched detections (Moderate)."""
    ious, errs, tp, n_gt = [], [], 0, 0
    for cls in CLASSES:
        r = evaluate_kitti(records, classes=(cls,), iou_thresh=0.5, difficulty=1)[cls]
        ious += r['match_iou']; errs += r['center_err']
        tp += r['tp']; n_gt += r['n_gt']
    return {
        'median_iou': round(float(np.median(ious)), 3) if ious else 0.0,
        'median_center_err_m': round(float(np.median(errs)), 3) if errs else 0.0,
        'recall': round(tp / n_gt, 3) if n_gt else 0.0,
        'tp': tp, 'n_gt': n_gt,
    }


def save_pr_curves(curves, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    colors = {'Car': '#34d399', 'Pedestrian': '#fb923c', 'Cyclist': '#60a5fa'}
    for cls in CLASSES:
        r = curves.get(cls)
        if not r or r['n_gt'] == 0 or len(r['recall']) < 2:
            continue
        ax.plot(r['recall'], r['precision'], color=colors[cls], linewidth=2,
                label=f"{cls} @IoU{STD_IOU[cls]} (AP={r['ap']:.3f})")
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("BEV Precision-Recall (Moderate)")
    ax.grid(alpha=0.3); ax.legend(loc='lower left')
    fig.tight_layout()
    path = os.path.join(out_dir, 'assets', 'pr_curves.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path); plt.close(fig)
    return path


def write_markdown(mat, loc, n_frames, out_dir):
    lines = [f"# Aggregate evaluation ({n_frames} KITTI frames)\n",
             "Bird's-eye-view Average Precision (VOC all-point AP), KITTI protocol "
             "(neighbour classes & DontCare regions ignored). IoU thresholds: "
             "Car 0.7, Pedestrian/Cyclist 0.5.\n",
             "| Class | IoU | Easy | Moderate | Hard | # GT (Mod) |",
             "|-------|:---:|-----:|---------:|-----:|-----------:|"]
    for cls in CLASSES:
        row = [cls, f"{STD_IOU[cls]}"]
        for name in DIFF_NAMES:
            row.append(f"{mat[cls][name]['ap']:.3f}")
        row.append(str(mat[cls]['Moderate']['n_gt']))
        lines.append("| " + " | ".join(row) + " |")
    map_mod = np.mean([mat[c]['Moderate']['ap'] for c in CLASSES])
    lines.append(f"\n**mAP (Moderate): {map_mod:.3f}**\n")
    lines.append("## Localization quality (matched detections, Moderate, IoU≥0.5)\n")
    lines.append(f"- Median BEV IoU: **{loc['median_iou']}**")
    lines.append(f"- Median center error: **{loc['median_center_err_m']} m**")
    lines.append(f"- Recall: **{loc['recall']}** ({loc['tp']}/{loc['n_gt']})\n")
    md = "\n".join(lines) + "\n"
    path = os.path.join(out_dir, 'EVALUATION.md')
    with open(path, 'w') as f:
        f.write(md)
    return path, md


def main():
    ap = argparse.ArgumentParser(description="KITTI-style BEV-AP evaluation")
    ap.add_argument('--data_dir', default='data/kitti')
    ap.add_argument('--out', default='docs')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--conf', type=float, default=0.2)
    ap.add_argument('--weights', default='yolov8s.pt')
    ap.add_argument('--head', choices=['geometric', 'learned'], default='geometric')
    args = ap.parse_args()

    pipe = FusionPipeline(args.data_dir, weights=args.weights, conf=args.conf,
                          head=args.head)
    frames = pipe.list_frames()
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        print("No frames found. Run scripts/download_subset.py first.")
        sys.exit(1)

    print(f"Evaluating on {len(frames)} frames (conf={args.conf})...")
    records = build_records(pipe, frames)
    mat, curves = ap_matrix(records)
    loc = localization_summary(records)

    curve_path = save_pr_curves(curves, args.out)
    md_path, md = write_markdown(mat, loc, len(frames), args.out)
    dump = {cls: {name: {k: v for k, v in mat[cls][name].items()
                         if k not in ('recall', 'precision', 'match_iou', 'center_err')}
                  for name in DIFF_NAMES} for cls in CLASSES}
    dump['localization'] = loc
    with open(os.path.join(args.out, 'evaluation.json'), 'w') as f:
        json.dump(dump, f, indent=2)

    print("\n" + md)
    print(f"Saved: {md_path}, {curve_path}, {args.out}/evaluation.json")


if __name__ == '__main__':
    main()
