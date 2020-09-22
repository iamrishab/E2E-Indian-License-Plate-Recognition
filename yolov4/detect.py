import os
import cv2

from detector import get_detections, initDetector
from pdb import set_trace

os.makedirs('results', exist_ok=True)


if __name__ == "__main__":
    # img_dir = '/home/rishab/lpr_dataset/Data-Images/Cars'
    img_dir = '../upload'
    configPath = '/home/rishab/licenseplaterecognition/yolo/yolov4-lpr.cfg'
    weightPath = '/home/rishab/licenseplaterecognition/yolo/weights/yolov4-lpr_6000.weights'
    metaPath = '/home/rishab/licenseplaterecognition/yolo/labels.data'
    netMain, metaMain = initDetector(configPath, weightPath, metaPath)
    os.system(f"rm -rf {img_dir}/.ipynb**")
    for file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, file)
        bbs = get_detections(img_path, netMain, metaMain)
        print(file, bbs)
