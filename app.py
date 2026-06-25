"""
Flask web app for the LiDAR + RGB late-fusion 3D detection pipeline.

Run:
    python app.py
    # then open http://localhost:5000

The detector/model is loaded once and reused. Each frame is processed on
demand; all visualizations are returned to the browser as base64 PNGs.
"""

import threading

import cv2
from flask import Flask, Response, abort, jsonify, render_template

from src.pipeline import FusionPipeline, detections_to_kitti_lines

app = Flask(__name__)

DATA_DIR = 'data/kitti'
_pipeline = FusionPipeline(data_dir=DATA_DIR, conf=0.35)
_lock = threading.Lock()  # YOLO/torch inference is not thread-safe under Flask

# In-memory cache of the most recently rendered PNGs: {(idx, name): png_bytes}.
_image_cache = {}


def _encode_png(img_bgr):
    ok, buf = cv2.imencode('.png', img_bgr)
    return buf.tobytes() if ok else None


def _serialize_detection(d):
    return {
        'kitti_class': d['kitti_class'],
        'coco_class': d.get('coco_class'),
        'score': round(d['score'], 3),
        'bbox': [round(float(x), 1) for x in d['bbox']],
        'depth': round(d.get('depth', 0.0), 1),
        'dimensions_hwl': [round(float(x), 2) for x in d['dimensions']],
        'location': [round(float(x), 2) for x in d['location']],
        'rotation_y': round(d['rotation_y'], 3),
        'n_points': d['n_points'],
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/frames')
def api_frames():
    frames = _pipeline.list_frames()
    info = []
    for idx in frames:
        info.append({
            'idx': idx,
            'has_labels': _pipeline.loader.has_labels(idx),
        })
    return jsonify({'frames': info})


@app.route('/api/image/<int:idx>/<name>.png')
def api_image(idx, name):
    png = _image_cache.get((idx, name))
    if png is None:
        abort(404)
    return Response(png, mimetype='image/png')


@app.route('/api/run/<int:idx>')
def api_run(idx):
    with _lock:
        res = _pipeline.run(idx, render=True)
        for name, img in res['images'].items():
            _image_cache[(idx, name)] = _encode_png(img)

    # Cache-busting token so the browser refetches after each run.
    images = {name: f'/api/image/{idx}/{name}.png?v={len(_image_cache)}'
              for name in res['images']}
    detections = [_serialize_detection(d) for d in res['detections_3d']]

    gt = [{'type': o['type'],
           'location': [round(float(x), 2) for x in o['location']],
           'dimensions': [round(float(x), 2) for x in o['dimensions']],
           'rotation_y': round(o['rotation_y'], 3)}
          for o in res['gt_objects'] if o['type'] != 'DontCare']

    return jsonify({
        'idx': idx,
        'n_lidar': res['n_lidar'],
        'image_shape': list(res['image_shape']),
        'n_2d': len(res['detections_2d']),
        'n_3d': len(res['detections_3d']),
        'images': images,
        'detections': detections,
        'gt_objects': gt,
        'metrics': res['metrics'],
        'kitti_lines': detections_to_kitti_lines(res['detections_3d']),
    })


if __name__ == '__main__':
    print("Loading pipeline... open http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
