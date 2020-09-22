import os
import cv2
import numpy as np
import requests
import uuid
from flask import Flask, request, json
from flask_cors import CORS
from flask_restplus import Api, Resource, reqparse
from werkzeug.datastructures import FileStorage
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from LPRv1.detect_license_plate import CarNumberPlateDetectorv1
from LPRv1.recognize_text_from_license_plate import TesserectRecognizeText

UPLOAD_IMAGE_FOLDER = 'upload'
os.makedirs(UPLOAD_IMAGE_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

api = Api(app, version='0.1', title="ONEBCG LPR", validate=False, description='LPR')
ns = api.namespace('pipeline', description='API Operations')

upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)

lprv1 = CarNumberPlateDetectorv1()
trt = TesserectRecognizeText()


@ns.route('/')
@api.expect(upload_parser)
class Upload(Resource):
    @api.response(200, 'response success.')
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        file_ext = uploaded_file.filename.split('.')[1]
        filename = str(uuid.uuid1()) + f'.{file_ext}'
        uploaded_file.save(os.path.join(UPLOAD_IMAGE_FOLDER, filename))
        file_path = os.path.join(UPLOAD_IMAGE_FOLDER, filename)
        img_bgr = cv2.imread(file_path)
        
        bbs = lprv1.detect(img_bgr)
        result = {}
        
        if len(bbs) == 1:
            xmin, ymin, xmax, ymax = bbs[0]
            cropped_image = img_bgr[ymin:ymax, xmin:xmax, :]
            text = trt.recognize_text(cropped_image)
            result = {'response': {f'{text}': [xmin, ymin, xmax, ymax]}}
        os.remove(file_path)
        return result, 200

    
if __name__ == '__main__':
    app.run('0.0.0.0', port=6500, debug=False, threaded=True)