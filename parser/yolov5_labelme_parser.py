import os
import io
import json
import cv2
import glob
import uuid
import base64
import numpy as np
import multiprocessing as mp

from pdb import set_trace

SUPPORTED_IMG_FORMAT = ['.png', '.jpg', '.jpeg']

gt_dir = '/home/rishab/lpr_dataset/labelled/combined'
# directory path to parsed ground truth to yolo format
save_dir = '/home/rishab/lpr_dataset/labelled/labels_yolov5'

train_split = 0.95 # train split
validation_split = 0.05 # validation split
workers = mp.cpu_count()


def stringToImage(base64_string):
    # reconstruct image as an numpy array
    decoded_data = base64.b64decode(base64_string)
    np_data = np.frombuffer(decoded_data, np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)


def process_file(gt_path):
    name_prefix = os.path.splitext(os.path.basename(gt_path))[0]
    uid = str(uuid.uuid4())
    # if name_prefix in img_prefixes:
    with open(gt_path) as fr:
        annotations = json.load(fr)
    if annotations['version'] == '4.2.10':
        height = annotations['imageHeight']
        width = annotations['imageWidth']
        image = stringToImage(annotations['imageData'])
        image_prefix, img_ext = os.path.splitext(os.path.basename(annotations['imagePath']))
        assert int(height) == image.shape[0]
        assert int(width) == image.shape[1]
        for annotation in annotations['shapes']:
            points_x = sorted(annotation['points'], key=lambda x: x[0])
            points_y = sorted(annotation['points'], key=lambda y: y[1])

            xmin_norm = max(0, int(points_x[0][0])) / width
            ymin_norm = max(0, int(points_y[0][1])) / height
            xmax_norm = min(width, int(points_x[-1][0])) / width
            ymax_norm = min(height, int(points_y[-1][1])) / height

            bb_width_norm = xmax_norm - xmin_norm
            bb_height_norm = ymax_norm - ymin_norm
            xcenter_norm = xmin_norm + bb_width_norm / 2
            ycenter_norm = ymin_norm + bb_height_norm / 2
            
            label_path = f"{os.path.join(save_dir, f'{uid}.txt')}"
            img_path = f"{os.path.join(save_dir, f'{uid}{img_ext}')}"
            with open(label_path, 'a+') as fa:
                fa.write(f"0 {xcenter_norm} {ycenter_norm} {bb_width_norm} {bb_height_norm}")
        cv2.imwrite(img_path, image)
        return img_path, label_path
    return '', ''


def parser():
    gt_paths = glob.glob(f"{gt_dir}/*.json")
        
    pool = mp.Pool(processes=workers)   
    results = pool.imap_unordered(process_file, gt_paths)
    
    total_imgs = []
    total_labels = []
    for result in results:
        img_path, label_path = result
        if img_path and label_path:
            total_imgs.append(img_path)
            total_labels.append(label_path)
        
    total_num = len(total_imgs) or len(total_labels)
    
    num_train = int(total_num * train_split)
    num_val = int(total_num * validation_split)

    img_path_train = total_imgs[:num_train]
    img_path_val = total_imgs[num_train:]
    
    labels_path_train = total_labels[:num_train]
    labels_path_val = total_labels[num_train:]

    os.makedirs(os.path.join(save_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'labels', 'train'), exist_ok=True)
    for img_path, label_path in zip(img_path_train, labels_path_train):
        os.rename(img_path, os.path.join(save_dir, 'images', 'train', os.path.basename(img_path)))
        os.rename(label_path, os.path.join(save_dir, 'labels', 'train', os.path.basename(label_path)))

    os.makedirs(os.path.join(save_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'labels', 'val'), exist_ok=True)
    for img_path, label_path in zip(img_path_val, labels_path_val):
        os.rename(img_path, os.path.join(save_dir, 'images', 'val', os.path.basename(img_path)))
        os.rename(label_path, os.path.join(save_dir, 'labels', 'val', os.path.basename(label_path)))

    
if __name__ == '__main__':
    os.makedirs(save_dir, exist_ok=True)
    os.system(f"rm -rf {save_dir}/*")
    parser()