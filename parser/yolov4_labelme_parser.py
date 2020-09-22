import os
import io
import json
import cv2
import glob
import base64
import numpy as np

from pdb import set_trace

SUPPORTED_IMG_FORMAT = ['.png', '.jpg', '.jpeg']

gt_dir = '/home/rishab/lpr_dataset/labelled/combined'
# directory path to parsed ground truth to yolo format
save_dir = '/home/rishab/lpr_dataset/labelled/labels_yolo'
# path to train labels
path_train = '/home/rishab/licenseplaterecognition/yolo/train.txt'
# path to validation labels
path_val = '/home/rishab/licenseplaterecognition/yolo/val.txt'
# path to test labels
path_test = '/home/rishab/licenseplaterecognition/yolo/test.txt'
label_names = '/home/rishab/licenseplaterecognition/yolo/labels.names'
label_data = '/home/rishab/licenseplaterecognition/yolo/labels.data'
weights = '/home/rishab/licenseplaterecognition/yolo/weights'
train_split = 0.9 # train split
validation_split = 0.05 # validation split
test_split = 0.05 # test split


def stringToImage(base64_string):
    # reconstruct image as an numpy array
    decoded_data = base64.b64decode(base64_string)
    np_data = np.frombuffer(decoded_data, np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)


def parser():
    gt_paths = glob.glob(f"{gt_dir}/*.json")
    # img_prefixes = [os.path.splitext(file)[0] for file in os.listdir(img_dir) if os.path.splitext(file)[-1].lower() in SUPPORTED_IMG_FORMAT]
    total_imgs = []
    count = 0
    for gt_path in gt_paths:
        name_prefix = os.path.splitext(os.path.basename(gt_path))[0]
        # if name_prefix in img_prefixes:
        with open(gt_path) as fr:
            annotations = json.load(fr)
        if annotations['version'] == '4.2.10':
            height = annotations['imageHeight']
            width = annotations['imageWidth']
            image = stringToImage(annotations['imageData'])
            image_prefix, ext = os.path.splitext(os.path.basename(annotations['imagePath']))
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

                with open(f"{os.path.join(save_dir, f'img_{count}.txt')}", 'a+') as fa:
                    fa.write(f"0 {xcenter_norm} {ycenter_norm} {bb_width_norm} {bb_height_norm}")
            cv2.imwrite(f"{os.path.join(save_dir, f'img_{count}{ext}')}", image)
            total_imgs.append(f"{os.path.join(save_dir, f'img_{count}{ext}')}")
            count += 1
    total_num_imgs = len(total_imgs)

    num_img_train = int(total_num_imgs * train_split)
    num_img_val = int(total_num_imgs * validation_split)
    num_img_test = int(total_num_imgs * test_split)

    img_path_train = total_imgs[:num_img_train]
    img_path_val = total_imgs[num_img_train:num_img_train+num_img_val]
    img_path_test = total_imgs[num_img_train+num_img_val:]

#     os.makedirs(os.path.dirname(os.path.abspath(path_train)), exist_ok=True)
#     if os.path.isdir(path_train):
#         path_train = os.path.join(path_train, 'train.txt')
    with open(path_train, 'w') as fw:
        for img_path in img_path_train:
            fw.write(os.path.abspath(img_path)+'\n')

#     os.makedirs(os.path.dirname(os.path.abspath(path_val)), exist_ok=True)
#     if os.path.isdir(path_val):
#         path_val = os.path.join(path_val, 'val.txt')
    with open(path_val, 'w') as fw:
        for img_path in img_path_val:
            fw.write(os.path.abspath(img_path)+'\n')

#     os.makedirs(os.path.dirname(os.path.abspath(path_test)), exist_ok=True)
#     if os.path.isdir(path_test):
#         path_test = os.path.join(path_test, 'test.txt')
    with open(path_test, 'w') as fw:
        for img_path in img_path_test:
            fw.write(os.path.abspath(img_path)+'\n')
    
#     os.makedirs(os.path.dirname(os.path.abspath(label_names)), exist_ok=True)
#     if os.path.isdir(label_names):
#         label_names = os.path.join(label_names, 'labels.names')
    with open(label_names, 'w') as fw:
        fw.write('licenseplate \n')
            
    os.makedirs(weights, exist_ok=True)
    with open(label_data, 'w') as fw:
        fw.write(f"classes = 1 \n")
        fw.write(f"train = {os.path.abspath(path_train)}\n")
        fw.write(f"valid = {os.path.abspath(path_val)}\n")
        fw.write(f"names = {os.path.abspath(label_names)}\n")
        fw.write(f"backup = {os.path.abspath(weights)}\n")
        # fw.write(f"eval = coco\n")
    
    
if __name__ == '__main__':
    # img_dir = "/home/rishab/lpr_dataset/labelled/combined"
    os.makedirs(save_dir, exist_ok=True)
    os.system(f"rm -rf {save_dir}/*")
    parser()