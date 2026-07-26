"""
Download one KITTI *tracking* sequence (consecutive frames) via partial-zip
extraction, for the multi-object tracking demo. ~360 MB for a 154-frame
sequence instead of the full ~37 GB velodyne archive.

Usage:
    python scripts/download_tracking.py --seq 0000 --out data/kitti_tracking
"""

import argparse
import os
import sys

BUCKET = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
ARCHIVES = {
    'velodyne': (f"{BUCKET}/data_tracking_velodyne.zip", 'bin', True),
    'image_02': (f"{BUCKET}/data_tracking_image_2.zip", 'png', True),
    'calib':    (f"{BUCKET}/data_tracking_calib.zip", 'txt', False),
    'label_02': (f"{BUCKET}/data_tracking_label_2.zip", 'txt', False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='0000')
    ap.add_argument('--out', default='data/kitti_tracking')
    ap.add_argument('--split', default='training')
    args = ap.parse_args()

    from remotezip import RemoteZip

    total = 0
    for sub, (url, ext, per_frame) in ARCHIVES.items():
        with RemoteZip(url) as z:
            if per_frame:
                out_dir = os.path.join(args.out, args.split, sub, args.seq)
                os.makedirs(out_dir, exist_ok=True)
                members = sorted(n for n in z.namelist()
                                 if n.startswith(f"{args.split}/{sub}/{args.seq}/")
                                 and n.endswith(f".{ext}"))
                print(f"[{sub}] {len(members)} frames...")
                for i, m in enumerate(members, 1):
                    dest = os.path.join(out_dir, os.path.basename(m))
                    if os.path.exists(dest) and os.path.getsize(dest) > 0:
                        continue
                    data = z.read(m)
                    open(dest, 'wb').write(data)
                    total += len(data)
                    if i % 25 == 0 or i == len(members):
                        print(f"  {i}/{len(members)} (~{total/1e6:.0f} MB)")
            else:
                out_dir = os.path.join(args.out, args.split, sub)
                os.makedirs(out_dir, exist_ok=True)
                member = f"{args.split}/{sub}/{args.seq}.{ext}"
                dest = os.path.join(out_dir, f"{args.seq}.{ext}")
                data = z.read(member)
                open(dest, 'wb').write(data)
                total += len(data)
                print(f"[{sub}] {args.seq}.{ext} ({len(data)} bytes)")

    print(f"\nDone. ~{total/1e6:.0f} MB into '{args.out}/{args.split}' (seq {args.seq}).")


if __name__ == '__main__':
    main()
