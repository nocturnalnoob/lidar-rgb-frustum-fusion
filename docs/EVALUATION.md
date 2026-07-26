# Aggregate evaluation (200 KITTI frames)

Bird's-eye-view Average Precision (VOC all-point AP), KITTI protocol (neighbour classes & DontCare regions ignored). IoU thresholds: Car 0.7, Pedestrian/Cyclist 0.5.

| Class | IoU | Easy | Moderate | Hard | # GT (Mod) |
|-------|:---:|-----:|---------:|-----:|-----------:|
| Car | 0.7 | 0.188 | 0.108 | 0.090 | 420 |
| Pedestrian | 0.5 | 0.353 | 0.305 | 0.267 | 84 |
| Cyclist | 0.5 | 0.222 | 0.194 | 0.212 | 31 |

**mAP (Moderate): 0.202**

## Localization quality (matched detections, Moderate, IoU≥0.5)

- Median BEV IoU: **0.753**
- Median center error: **0.159 m**
- Recall: **0.323** (173/535)

