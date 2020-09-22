from flask import Flask, Response, request, redirect, jsonify
import json

from detector import *


app = Flask(__name__)


weightPath = 'weights/best.pt'
model = load_model(weightPath)

            
@app.route("/yolov5",methods=["GET", "POST"])
def main():
    img_path = request.form['path']
    bbs = detect(img_path, model)
    return json.dumps(bbs)

            
if __name__ == '__main__':
    app.run('0.0.0.0', '8003', debug=False, threaded=True)

