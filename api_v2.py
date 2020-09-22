import os
import cv2
import uuid
import base64
import shutil
import requests
import numpy as np

from flask import Flask, request, json
from flask_cors import CORS
from flask_restplus import Api, Resource, reqparse
from werkzeug.datastructures import FileStorage
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from utils.preprocess import *
from config import *

UPLOAD_IMAGE_FOLDER = 'upload'
os.makedirs(UPLOAD_IMAGE_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

api = Api(app, version='0.1', title="ONEBCG ALPR", validate=False, description='ALPR')
ns = api.namespace('alpr', description='')

upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)


@ns.route('/')
@api.expect(upload_parser)
class Upload(Resource):
    @api.response(200, 'response success.')
    def post(self):
        try:
            args = upload_parser.parse_args()
            uploaded_file = args['file']  # This is FileStorage instance
            # import pdb; pdb.set_trace()
            # file_ext = uploaded_file.filename.split('.')[1]
            session_id = str(uuid.uuid1())
            # filename = f'{session_id}.{file_ext}'
            filename = f'{session_id}.png'
            uploaded_file.save(os.path.join(UPLOAD_IMAGE_FOLDER, filename))
            file_path = os.path.abspath(os.path.join(UPLOAD_IMAGE_FOLDER, filename))
            save_cropped_img_dir = os.path.abspath(f"{UPLOAD_IMAGE_FOLDER}/{session_id}")
            os.makedirs(save_cropped_img_dir)
            save_cropped_img_path = os.path.abspath(f"{save_cropped_img_dir}/cropped.png")

            # img_str = cv2.imencode('.jpg', self.frame)[1].tostring()
            img_bgr = cv2.imread(file_path)
            # preprocessing step
            img_bgr, _, _ = automatic_brightness_and_contrast(img_bgr, clip_hist_percent=5)
            img_bgr = sharpen_image(img_bgr)
            # bbs = requests.post('http://0.0.0.0:8001/yolov4', data = {'path': file_path}).json()
            bbs = requests.post(f'http://{DETECTION_IP}:{DETECTION_PORT}/yolov5', data = {'path': file_path}).json()
            # bbs = get_bb(file_path)
            result = {}
            if len(bbs) == 1:
                xmin, ymin, xmax, ymax = bbs[0]
                cropped_image = img_bgr[ymin:ymax, xmin:xmax, :]
                cv2.imwrite(save_cropped_img_path, cropped_image)
                text = requests.post(f'http://{RECOGNITION_IP}:{RECOGNITION_PORT}/deeptext', data = {'path': save_cropped_img_path}).json()
                result = {'numberPlate': f'{text[0]}', 'boundingBox': [xmin, ymin, xmax, ymax], 'cropped': im2str(cropped_image)}
                # cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # cv2.putText(img_bgr, text[0], (xmin-10, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                # cv2.imwrite(file_path, img_bgr)
            os.remove(file_path)
            shutil.rmtree(save_cropped_img_dir)
        except:
            pass
        finally:
            pass
        return result

    
if __name__ == '__main__':
    app.run(SWAGGER_IP, port=SWAGGER_PORT, debug=False, threaded=True)