"""
Run the full LiDAR+RGB late-fusion 3D detection pipeline on one or more frames
and save all visualizations + a KITTI-format prediction file.

Usage:
    python scripts/run_fusion.py --data_dir data/kitti --idx 0
    python scripts/run_fusion.py --data_dir data/kitti --all --out outputs
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import FusionPipeline, detections_to_kitti_lines


def process(pipe, idx, out_dir):
    res = pipe.run(idx, render=True)
    prefix = f"{idx:06d}"
    paths = FusionPipeline.save_images(res['images'], out_dir, prefix)

    # Save KITTI-format predictions.
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, f"{prefix}_pred.txt")
    with open(pred_path, 'w') as f:
        f.write('\n'.join(detections_to_kitti_lines(res['detections_3d'])))

    print(f"\nFrame {prefix}: {res['n_lidar']} LiDAR pts | "
          f"{len(res['detections_2d'])} 2D dets -> "
          f"{len(res['detections_3d'])} fused 3D boxes")
    for d in res['detections_3d']:
        print(f"   - {d['kitti_class']:<11} depth={d['depth']:5.1f}m  "
              f"dims(hwl)={d['dimensions'].round(2)}  pts={d['n_points']}")
    if res['metrics']:
        m = res['metrics']['overall']
        print(f"   BEV eval @IoU{res['metrics']['iou_thresh']}: "
              f"precision={m['precision']} recall={m['recall']} "
              f"meanIoU={m['mean_iou']}")
    print(f"   saved -> {out_dir}/ ({', '.join(os.path.basename(p) for p in paths.values())})")


def main():
    ap = argparse.ArgumentParser(description="LiDAR+RGB fusion 3D detection")
    ap.add_argument('--data_dir', default='data/kitti')
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--all', action='store_true', help='process every frame found')
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--weights', default='yolov8s.pt')
    ap.add_argument('--conf', type=float, default=0.35)
    args = ap.parse_args()

    pipe = FusionPipeline(args.data_dir, weights=args.weights, conf=args.conf)
    frames = pipe.list_frames() if args.all else [args.idx]
    if not frames:
        print("No frames found. Run: python scripts/download_sample.py")
        sys.exit(1)

    for idx in frames:
        process(pipe, idx, args.out)


if __name__ == '__main__':
    main()
