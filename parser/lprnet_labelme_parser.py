import os
import io
import json
import cv2
import glob
import base64
import numpy as np

from pdb import set_trace

SUPPORTED_IMG_FORMAT = ['.png', '.jpg', '.jpeg']

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
         'U', 'V', 'W', 'X', 'Y', 'Z'
         ]

def normalize_text(txt):
    # replace special characters
    for x in list(txt):
        if x not in CHARS:
            txt = txt.replace(x, '') 
    return txt
        

def stringToImage(base64_string):
    # reconstruct image as an numpy array
    decoded_data = base64.b64decode(base64_string)
    np_data = np.frombuffer(decoded_data, np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)


def parser(gt_dir, save_dir):
    gt_paths = glob.glob(f"{gt_dir}/*.json")
    # img_prefixes = [os.path.splitext(file)[0] for file in os.listdir(img_dir) if os.path.splitext(file)[-1].lower() in SUPPORTED_IMG_FORMAT]
    for gt_path in gt_paths:
        name_prefix = os.path.splitext(os.path.basename(gt_path))[0]
        # if name_prefix in img_prefixes:
        with open(gt_path) as fr:
            annotations = json.load(fr)
        if annotations['version'] == '4.2.10':
            height = annotations['imageHeight']
            width = annotations['imageWidth']
            image = stringToImage(annotations['imageData'])
            assert int(height) == image.shape[0]
            assert int(width) == image.shape[1]
            for annotation in annotations['shapes']:
                if '%' in annotation['label'] or 'TEMP' in annotation['label']:
                    print('Not parsing TEMP or Double lines Number Plates')
                    continue
                imgname = normalize_text(annotation['label'])
                points_x = sorted(annotation['points'], key=lambda x: x[0])
                points_y = sorted(annotation['points'], key=lambda y: y[1])

                xmin, ymin = max(0, int(points_x[0][0])), max(0, int(points_y[0][1]))
                xmax, ymax = min(width, int(points_x[-1][0])), min(height, int(points_y[-1][1]))

                cv2.imwrite(f"{os.path.join(save_dir, f'{imgname}.jpg')}", image[ymin:ymax, xmin:xmax, :])
    
if __name__ == '__main__':
    # img_dir = "/home/rishab/lpr_dataset/labelled/combined"
    gt_dir = "/home/rishab/lpr_dataset/labelled/combined"
    save_dir = "/home/rishab/licenseplaterecognition/LPRNet/dataset/combined"
    os.makedirs(save_dir, exist_ok=True)
    parser(gt_dir, save_dir)