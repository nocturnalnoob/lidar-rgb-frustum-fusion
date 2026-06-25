"""
RGB 2D object detector built on a pretrained YOLOv8 model (Ultralytics).

No training is required: we use COCO-pretrained weights and remap the relevant
COCO classes onto the KITTI taxonomy (Car / Pedestrian / Cyclist). The detector
runs comfortably on CPU with the nano weights (`yolov8n.pt`, ~6 MB).
"""

import numpy as np

# COCO class name -> KITTI class name. Anything not listed is discarded.
COCO_TO_KITTI = {
    'car': 'Car',
    'truck': 'Car',
    'bus': 'Car',
    'person': 'Pedestrian',
    'bicycle': 'Cyclist',
    'motorcycle': 'Cyclist',
}

# Colors (BGR) used consistently across all visualizations.
KITTI_COLORS = {
    'Car': (0, 200, 0),
    'Pedestrian': (0, 140, 255),
    'Cyclist': (255, 80, 0),
}


class YoloDetector:
    def __init__(self, weights='yolov8n.pt', conf=0.35, device='cpu'):
        # Imported lazily so the rest of the package works without ultralytics.
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        self.names = self.model.names  # {class_id: name}

    def detect(self, image_bgr):
        """
        Run 2D detection on a BGR image.

        Returns a list of detections, each a dict with:
            bbox  : np.array([x1, y1, x2, y2])
            score : float
            kitti_class : str  (Car / Pedestrian / Cyclist)
            coco_class  : str
        """
        results = self.model.predict(
            image_bgr, conf=self.conf, device=self.device, verbose=False
        )[0]

        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            coco_name = self.names.get(cls_id, str(cls_id))
            kitti_class = COCO_TO_KITTI.get(coco_name)
            if kitti_class is None:
                continue
            xyxy = box.xyxy[0].cpu().numpy().astype(float)
            detections.append({
                'bbox': xyxy,
                'score': float(box.conf[0]),
                'kitti_class': kitti_class,
                'coco_class': coco_name,
            })
        return detections
