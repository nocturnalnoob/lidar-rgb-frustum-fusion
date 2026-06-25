"""
Fetch a few real KITTI sample frames (image + LiDAR + calibration + labels) so
the pipeline can be run end-to-end without downloading the full ~5 GB dataset.

Usage:
    python scripts/download_sample.py --out data/kitti

Frames are pulled from the public `kuixu/kitti_object_vis` mirror. For the full
dataset, see the instructions printed at the end / in the README.
"""

import argparse
import os
import sys
import urllib.request

BASE = ("https://raw.githubusercontent.com/kuixu/kitti_object_vis/"
        "master/data/object/training")
SUBDIRS = {'calib': 'txt', 'image_2': 'png', 'velodyne': 'bin', 'label_2': 'txt'}
# Frames known to exist in the mirror.
SAMPLE_FRAMES = ['000000', '000001', '000002']


def download(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent': 'curl/8'})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, 'wb') as f:
        f.write(r.read())


def main():
    ap = argparse.ArgumentParser(description="Download sample KITTI frames")
    ap.add_argument('--out', default='data/kitti', help='dataset root')
    ap.add_argument('--split', default='training')
    args = ap.parse_args()

    n_ok = 0
    for frame in SAMPLE_FRAMES:
        for sub, ext in SUBDIRS.items():
            out_dir = os.path.join(args.out, args.split, sub)
            os.makedirs(out_dir, exist_ok=True)
            dest = os.path.join(out_dir, f"{frame}.{ext}")
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                print(f"  skip  {sub}/{frame}.{ext} (exists)")
                n_ok += 1
                continue
            url = f"{BASE}/{sub}/{frame}.{ext}"
            try:
                download(url, dest)
                print(f"  ok    {sub}/{frame}.{ext} "
                      f"({os.path.getsize(dest)} bytes)")
                n_ok += 1
            except Exception as e:
                print(f"  FAIL  {sub}/{frame}.{ext}: {e}", file=sys.stderr)

    print(f"\nDownloaded/verified {n_ok} files into '{args.out}'.")
    print("\nTo use the FULL KITTI 3D object detection dataset (~5 GB / 7481 "
          "frames):")
    print("  1. Register at https://www.cvlibs.net/datasets/kitti/eval_object.php")
    print("  2. Download: left color images, Velodyne point clouds, "
          "camera calibration, and training labels.")
    print(f"  3. Unzip so the structure matches '{args.out}/training/"
          "{calib,image_2,velodyne,label_2}/'.")


if __name__ == '__main__':
    main()
