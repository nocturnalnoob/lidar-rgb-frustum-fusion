import argparse
import sys
import os
import cv2

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.kitti_loader import KittiLoader
from src.calibration.project_lidar import project_velo_to_image, overlay_lidar_on_image

def main():
    parser = argparse.ArgumentParser(description="Visualize LiDAR projection onto RGB image")
    parser.add_argument('--data_dir', type=str, default='./data/kitti', help='Path to KITTI dataset root')
    parser.add_argument('--idx', type=int, default=0, help='Index of the frame to visualize')
    parser.add_argument('--max_depth', type=float, default=70.0, help='Max visualization depth (m)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save overlay to this path instead of opening a window')
    args = parser.parse_args()

    # Initialize loader
    loader = KittiLoader(args.data_dir, split='training')
    
    try:
        calib = loader.get_calib(args.idx)
        image = loader.get_image(args.idx)
        lidar = loader.get_lidar(args.idx)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure the KITTI dataset is placed in `--data_dir`.")
        print(f"Expected path: {os.path.join(args.data_dir, 'training')}")
        print("Required subfolders: calib/, image_2/, velodyne/")
        sys.exit(1)
        
    print(f"Loaded frame {args.idx:06d}:")
    print(f" - Image shape: {image.shape}")
    print(f" - LiDAR points: {lidar.shape[0]}")
    
    # Project LiDAR to 2D
    pts_2d, pts_cam_rect, mask = project_velo_to_image(lidar, calib)
    
    # Extract depth (Z coordinate in camera rect frame)
    depth = pts_cam_rect[:, 2]
    print(f" - Points projected in front of camera: {len(pts_2d)}")
    
    # Generate overlay
    img_overlay = overlay_lidar_on_image(image, pts_2d, depth, max_depth=args.max_depth)
    
    # Save to file, or display in a window when a GUI is available.
    if args.save:
        cv2.imwrite(args.save, img_overlay)
        print(f"Saved overlay to {args.save}")
        return
    try:
        window_name = f"LiDAR Projection (Frame {args.idx:06d})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img_overlay)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        fallback = f"projection_{args.idx:06d}.png"
        cv2.imwrite(fallback, img_overlay)
        print(f"No display available; saved overlay to {fallback}")

if __name__ == '__main__':
    main()
