# app.py
from flask import Flask, request, render_template, jsonify, send_file
import cv2
import numpy as np, os, io
from inference import run_yolo, run_change_detector, compute_damage_for_objects
from matplotlib import pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return '''
    <h2>Disaster Change Detection</h2>
    <form method="post" action="/detect" enctype="multipart/form-data">
      Previous image: <input type="file" name="img_old"><br>
      Current image: <input type="file" name="img_new"><br>
      <input type="submit" value="Analyze">
    </form>
    '''

@app.route('/detect', methods=['POST'])
def detect():
    f_old = request.files.get('img_old')
    f_new = request.files.get('img_new')
    if not f_old or not f_new:
        return "Please upload both images", 400
    # read images into OpenCV BGR
    old_bytes = f_old.read()
    new_bytes = f_new.read()
    old_arr = np.frombuffer(old_bytes, np.uint8)
    new_arr = np.frombuffer(new_bytes, np.uint8)
    img_old = cv2.imdecode(old_arr, cv2.IMREAD_COLOR)
    img_new = cv2.imdecode(new_arr, cv2.IMREAD_COLOR)

    change_mask = run_change_detector(img_old, img_new, thresh=0.35)  # tune threshold

    objs = run_yolo(img_new, conf=0.25)

   
    objs_with_damage = compute_damage_for_objects(objs, change_mask)
    vis = img_new.copy()
    red_overlay = vis.copy()
    red_overlay[change_mask>127] = (0,0,255)  # BGR: red
    alpha = 0.4
    vis = cv2.addWeighted(red_overlay, alpha, vis, 1-alpha, 0)
    for o in objs_with_damage:
        x1,y1,x2,y2 = o['bbox']
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        txt = f"{o['class']} {o['damage_pct']:.1f}%"
        cv2.putText(vis, txt, (max(0,x1), max(10,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    _, imbuf = cv2.imencode('.png', vis)
    img_bytes = imbuf.tobytes()


    response = {
        'objects': [{
            'class': o['class'],
            'score': o.get('score', None),
            'bbox': o['bbox'],
            'damage_pct': round(o['damage_pct'], 2)
        } for o in objs_with_damage],
    }
    return (send_file(io.BytesIO(img_bytes), mimetype='image/png'), 200, {'Content-Type': 'image/png'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
