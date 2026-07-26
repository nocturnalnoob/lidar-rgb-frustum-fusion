"""
RGB 2D object detector built on a pretrained YOLOv8 model (Ultralytics).

No training is required: we use COCO-pretrained weights and remap the relevant
COCO classes onto the KITTI taxonomy (Car / Pedestrian / Cyclist). The detector
runs comfortably on CPU with the nano weights (`yolov8n.pt`, ~6 MB).
"""

import numpy as np

# COCO vehicle classes -> KITTI Car.
COCO_CARS = {'car', 'truck', 'bus'}
# COCO two-wheelers. A KITTI "Cyclist" is a *person riding* one of these, so a
# bare bicycle/motorcycle is NOT emitted on its own (a parked bike is not a
# Cyclist and appears in no KITTI label). We emit Cyclist only when a person
# detection overlaps a two-wheeler; otherwise a person is a Pedestrian.
COCO_TWOWHEEL = {'bicycle', 'motorcycle'}

# Colors (BGR) used consistently across all visualizations.
KITTI_COLORS = {
    'Car': (0, 200, 0),
    'Pedestrian': (0, 140, 255),
    'Cyclist': (255, 80, 0),
}


def _overlap_ratio(bike, person):
    """Fraction of the bike box covered by the person box (intersection / bike area)."""
    ix1, iy1 = max(bike[0], person[0]), max(bike[1], person[1])
    ix2, iy2 = min(bike[2], person[2]), min(bike[3], person[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    bike_area = (bike[2] - bike[0]) * (bike[3] - bike[1])
    return inter / bike_area if bike_area > 0 else 0.0


class YoloDetector:
    def __init__(self, weights='yolov8s.pt', conf=0.35, device='cpu'):
        # Imported lazily so the rest of the package works without ultralytics.
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        self.names = self.model.names  # {class_id: name}

    def detect(self, image_bgr):
        """
        Run 2D detection on a BGR image and map to KITTI classes.

        Cyclist is formed by associating a `person` box with an overlapping
        two-wheeler box (their union); unmatched persons become Pedestrians and
        bare two-wheelers are dropped. Returns a list of dicts with:
            bbox  : np.array([x1, y1, x2, y2])
            score : float
            kitti_class : str  (Car / Pedestrian / Cyclist)
            coco_class  : str
        """
        results = self.model.predict(
            image_bgr, conf=self.conf, device=self.device, verbose=False
        )[0]
        if results.boxes is None:
            return []

        cars, persons, bikes = [], [], []
        for box in results.boxes:
            coco = self.names.get(int(box.cls[0]), '')
            xyxy = box.xyxy[0].cpu().numpy().astype(float)
            score = float(box.conf[0])
            if coco in COCO_CARS:
                cars.append((xyxy, score, coco))
            elif coco == 'person':
                persons.append([xyxy, score])
            elif coco in COCO_TWOWHEEL:
                bikes.append((xyxy, score, coco))

        detections = [{'bbox': b, 'score': s, 'kitti_class': 'Car', 'coco_class': c}
                      for b, s, c in cars]

        # Rider = person box overlapping a two-wheeler -> Cyclist (union box).
        used = set()
        for bxy, bsc, bcoco in bikes:
            best_pi, best_ov = -1, 0.30  # require >=30% of the bike covered
            for pi, (pxy, psc) in enumerate(persons):
                if pi in used:
                    continue
                ov = _overlap_ratio(bxy, pxy)
                if ov > best_ov:
                    best_ov, best_pi = ov, pi
            if best_pi >= 0:
                used.add(best_pi)
                pxy, psc = persons[best_pi]
                union = np.array([min(bxy[0], pxy[0]), min(bxy[1], pxy[1]),
                                  max(bxy[2], pxy[2]), max(bxy[3], pxy[3])])
                detections.append({'bbox': union, 'score': max(bsc, psc),
                                   'kitti_class': 'Cyclist', 'coco_class': bcoco})

        for pi, (pxy, psc) in enumerate(persons):
            if pi not in used:
                detections.append({'bbox': pxy, 'score': psc,
                                   'kitti_class': 'Pedestrian', 'coco_class': 'person'})
        return detections
