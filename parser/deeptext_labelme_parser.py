"""
    LMDB parser for deeptext
"""

import os
import io
import json
import cv2
import glob
import random
import base64
import shutil
import numpy as np

from pdb import set_trace

SUPPORTED_IMG_FORMAT = ['.png', '.jpg', '.jpeg']

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
         'U', 'V', 'W', 'X', 'Y', 'Z'
         ]

gt_dir = '/home/rishab/lpr_dataset/labelled/combined'
save_dir = '/home/rishab/lpr_dataset/labelled/deeptext'
train_split = 0.95 # train split
validation_split = 0.05 # validation split
# test_split = 0.05 # test split


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
    return cv2.imdecode(np_data, cv2.IMREAD_COLOR)


def parser(gt_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'tmp'), exist_ok=True)
    
    total_lmdb_list = []
    
    for gt_path in sorted(glob.glob(f"{gt_dir}/*.json"), key=lambda x: random.random()):
        name_prefix = os.path.splitext(os.path.basename(gt_path))[0]
        # if name_prefix in img_prefixes:
        with open(gt_path) as fr:
            annotations = json.load(fr)
        height = annotations['imageHeight']
        width = annotations['imageWidth']
        image = stringToImage(annotations['imageData'])
        if image is not None:
            assert int(height) == image.shape[0]
            assert int(width) == image.shape[1]
            for annotation in annotations['shapes']:
                if '%' in annotation['label']:
                    continue
                # label = normalize_text(annotation['label'])
                label = annotation['label'].replace(' ', '')

                points_x = sorted(annotation['points'], key=lambda x: x[0])
                points_y = sorted(annotation['points'], key=lambda y: y[1])

                xmin, ymin = max(0, int(points_x[0][0])), max(0, int(points_y[0][1]))
                xmax, ymax = min(width, int(points_x[-1][0])), min(height, int(points_y[-1][1]))

                crop_lp_save_path = f"{os.path.join(save_dir, 'tmp', f'{label}.jpg')}"
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
                cv2.imwrite(crop_lp_save_path, image[ymin:ymax, xmin:xmax, :])
                total_lmdb_list.append(f'{crop_lp_save_path}\t{label}')
    
    total_num_imgs = len(total_lmdb_list)

    num_img_train = int(total_num_imgs * train_split)
    num_img_val = int(total_num_imgs * validation_split)
#     num_img_test = int(total_num_imgs * test_split)

    train_list = total_lmdb_list[:num_img_train]
    val_list = total_lmdb_list[num_img_train:num_img_train+num_img_val]
#     test_list = total_lmdb_list[num_img_train+num_img_val:]
    
    save_train_dir = os.path.join(save_dir, 'train') 
    os.makedirs(f'{save_train_dir}/data', exist_ok=True)
    with open(os.path.join(save_train_dir, 'gt.txt'), 'w') as fw:
        for line in train_list:
            img_path, label = line.split('\t')
            if os.path.exists(img_path):
                new_img_path = os.path.join(save_train_dir, 'data', os.path.basename(img_path))
                os.rename(img_path, new_img_path)
                fw.write(f"{os.path.abspath(new_img_path)}\t{label}\n")
    
    save_val_dir = os.path.join(save_dir, 'val')
    os.makedirs(f'{save_val_dir}/data', exist_ok=True)
    with open(os.path.join(save_val_dir, 'gt.txt'), 'w') as fw:
        for line in val_list:
            img_path, label = line.split('\t')
            if os.path.exists(img_path):
                new_img_path = os.path.join(save_val_dir, 'data', os.path.basename(img_path))
                os.rename(img_path, new_img_path)
                fw.write(f"{os.path.abspath(new_img_path)}\t{label}\n")
    
#     save_test_dir = os.path.join(save_dir, 'test')
#     os.makedirs(f'{save_test_dir}/data', exist_ok=True)
#     with open(os.path.join(save_test_dir, 'gt.txt'), 'w') as fw:
#         for line in test_list:
#             img_path, label = line.split('\t')
#             if os.path.exists(img_path):
#                 new_img_path = os.path.join(save_test_dir, 'data', os.path.basename(img_path))
#                 os.rename(img_path, new_img_path)
#                 fw.write(f"{os.path.abspath(new_img_path)}\t{label}\n")
    
    shutil.rmtree(os.path.join(save_dir, 'tmp'))


if __name__ == '__main__':
    os.system(f'rm -rf {save_dir}/*')
    parser(gt_dir, save_dir)
    