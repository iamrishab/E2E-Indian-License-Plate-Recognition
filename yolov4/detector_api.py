import os
import cv2
from flask import Flask, Response, request, redirect, jsonify
import json

from detector import initDetector, get_detections


app = Flask(__name__)

configPath = 'yolov4-lpr.cfg'
weightPath = 'weights/yolov4-lpr_6000.weights'
metaPath = 'labels.data'
netMain, metaMain = initDetector(configPath, weightPath, metaPath)


# def get_bb(img_path):
#     img_path = os.path.join(img_dir, file)
#     bbs = get_detections(img_path, netMain, metaMain)
#     return bbs

            
@app.route("/yolov4",methods=["GET", "POST"])
def main():
    img_path = request.form['path']
    bbs = get_detections(img_path, netMain, metaMain)
    return json.dumps(bbs)

            
if __name__ == '__main__':
    app.run('0.0.0.0', '8001', debug=False, threaded=True)

