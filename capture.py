import os
import cv2
import uuid
import shutil
import requests
import numpy as np

from config import *
from utils.preprocess import *

UPLOAD_IMAGE_FOLDER = 'upload'
os.makedirs(UPLOAD_IMAGE_FOLDER, exist_ok=True)
    
    
def main():
    # reading camera feed
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not isinstance(VIDEO_SOURCE, str):
        # setting camera parameters
        cap.set(FRAME_WIDTH, cv2.CAP_PROP_FRAME_WIDTH)
        cap.set(FRAME_HEIGHT, cv2.CAP_PROP_FRAME_HEIGHT)

    # reading feed from camera
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        # close if the input source cannot fetch any frame
        if not ret:
            break

        img_str = cv2.imencode('.jpg', frame_bgr)[1].tostring()
        result = requests.post(f'http://{SWAGGER_IP}:{SWAGGER_PORT}/alpr/', files={'file': img_str}).json()
        if len(result) > 1:
            print(result['numberPlate'], result['boundingBox'])

#       # uncomment when using UI
#       cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#         cv2.putText(img_bgr, text[0], (xmin-10, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
#       cv2.imshow('frame', img_bgr)
#         if cv2.waitKey(1) == ord('q'):
#             break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()