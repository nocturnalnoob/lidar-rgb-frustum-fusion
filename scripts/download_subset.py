"""
Download a SUBSET of the full KITTI 3D object-detection dataset (a few hundred
frames) without pulling the entire ~41 GB.

The official images/velodyne archives are single monolithic zips, but S3 serves
them with HTTP range support, so `remotezip` can extract individual entries and
download only the bytes for the frames we ask for. ~200 frames is roughly 1 GB.

Usage:
    python scripts/download_subset.py --out data/kitti --n 200
"""

import argparse
import os
import sys

BUCKET = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
ARCHIVES = {
    'velodyne': f"{BUCKET}/data_object_velodyne.zip",
    'image_2': f"{BUCKET}/data_object_image_2.zip",
    'calib': f"{BUCKET}/data_object_calib.zip",
    'label_2': f"{BUCKET}/data_object_label_2.zip",
}
EXT = {'velodyne': 'bin', 'image_2': 'png', 'calib': 'txt', 'label_2': 'txt'}


def main():
    ap = argparse.ArgumentParser(description="Download a KITTI subset via partial zip extraction")
    ap.add_argument('--out', default='data/kitti')
    ap.add_argument('--n', type=int, default=200, help='number of training frames')
    ap.add_argument('--split', default='training')
    args = ap.parse_args()

    try:
        from remotezip import RemoteZip
    except ImportError:
        print("Missing dependency. Install it with:  pip install remotezip", file=sys.stderr)
        sys.exit(1)

    # Pick the frame ids from the velodyne archive's listing (authoritative).
    print("Reading archive index (velodyne)...")
    with RemoteZip(ARCHIVES['velodyne']) as z:
        ids = sorted(
            os.path.splitext(os.path.basename(n))[0]
            for n in z.namelist()
            if n.startswith(f"{args.split}/velodyne/") and n.endswith('.bin')
        )
    frame_ids = ids[:args.n]
    print(f"Selecting {len(frame_ids)} frames: {frame_ids[0]}..{frame_ids[-1]}")

    total_bytes = 0
    for sub, url in ARCHIVES.items():
        out_dir = os.path.join(args.out, args.split, sub)
        os.makedirs(out_dir, exist_ok=True)
        ext = EXT[sub]
        print(f"\n[{sub}] extracting {len(frame_ids)} files...")
        with RemoteZip(url) as z:
            for i, fid in enumerate(frame_ids, 1):
                dest = os.path.join(out_dir, f"{fid}.{ext}")
                if os.path.exists(dest) and os.path.getsize(dest) > 0:
                    continue
                member = f"{args.split}/{sub}/{fid}.{ext}"
                try:
                    data = z.read(member)
                    with open(dest, 'wb') as f:
                        f.write(data)
                    total_bytes += len(data)
                except Exception as e:
                    print(f"  FAIL {member}: {e}", file=sys.stderr)
                if i % 25 == 0 or i == len(frame_ids):
                    print(f"  {sub}: {i}/{len(frame_ids)}  (~{total_bytes/1e6:.0f} MB so far)")

    print(f"\nDone. Downloaded ~{total_bytes/1e6:.0f} MB into '{args.out}/{args.split}'.")
    print(f"Frames now available: {len(frame_ids)} (+ any previously downloaded).")


if __name__ == '__main__':
    main()
