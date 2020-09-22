from __future__ import print_function

import os
import sys
import cv2
import glob
import json
import numpy as np

import torch

from utils.detect import im_detect
from utils.utils import is_image, get_iou
from licenseplaterecognition.eval.script import master_evaluation


def calculate_precision_recall(tp, tn, fp, fn):
	"""
	precision-recall curves are appropriate for imbalanced datasets.
	"""
	delta = 1e-3
	precision = tp / (tp + fp + delta)
	recall = tp / (tp + fn + delta)
	f1 = (2 * precision * recall) / (precision + recall + delta)
	accuracy = (tp + tn) / (tp + tn + fp + fn + delta)
	return precision, recall, f1, accuracy


def parse_gt_txt(args):
    # removing unnecessary files and taking only the matching ones
    ims_list = sorted([file for file in os.listdir(args.test_img) if not file.startswith('.') and is_image(file)])
    img_list_prefix = [os.path.splitext(im)[0] for im in ims_list]
    
    gt_file_paths = glob.glob(f"{args.test_gt}/*.txt")
    parsed_gt_dir = os.path.join(os.getcwd(), 'gt')
    if len(os.listdir(parsed_gt_dir)):
        os.system(f'rm {parsed_gt_dir}/*')
    os.makedirs(parsed_gt_dir, exist_ok=True)
    gt_boxes = []
    
    for gt_file in gt_file_paths:
        gt_prefix = os.path.splitext(os.path.basename(gt_file))[0].split('gt_', 1)[-1]
        if gt_prefix in img_list_prefix:
            im_path = os.path.join(args.test_img, ims_list[img_list_prefix.index(gt_prefix)])
            src = cv2.imread(im_path, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            h, w = im.shape[:2]

            with open(gt_file) as fr:
                lines = fr.readlines()
                for line in lines:
                    line = line.strip()
                    if ',' in line and ' ' in line:
                        line = line.replace(' ', '')
                        sep, count_sep = ',', line.count(',')                    
                    elif ',' in line:
                        sep, count_sep = ',', line.count(',')
                    elif ' ' in line:
                        sep, count_sep = ' ', line.count(' ')
                    else:
                        raise Exception(line, 'Cannot find GT separator')

                    if count_sep >= 8:
                        line = line.split(sep, 8)
                    else:
                        line = line.split(sep, 4)

                    if len(line) == 5:
                        try:
                            x1, y1, x2, y2 = map(float, line[1:])
                        except ValueError:
                            x1, y1, x2, y2 = map(float, line[0:4])
                        except Exception:
                            continue
                        if x1 < x2 <= 1. and y1 < y2 <= 1.:
                            x1, y1, x2, y2 = x1*w, y1*h, x2*w, y2*h
                        gt_boxes.append(np.array([x1, y1, x2, y2], dtype=np.int32))
                    elif len(line) == 9:
                        try:
                            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line[-8:])
                        except ValueError:
                            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line[0:8])
                        except Exception:
                            continue
                        if x1 < x3 <= 1. and y1 < y3 <= 1.:
                            x1, y1, x2, y2, x3, y3, x4, y4 = x1*w, y1*h, x2*w, y2*h, x3*w, y3*h, x4*w, y4*h

                        gt_boxes.append(np.array([int(x1), int(y1), int(x3), int(y3)], dtype=np.int32))

                    with open(os.path.join(parsed_gt_dir, os.path.basename(gt_file)), 'a+') as fa:
                        fa.write(f'{int(x1)}, {int(y1)}, {int(x3)}, {int(y3)}, ###\n')
    if os.path.exists(os.path.join(os.getcwd(), 'gt.zip')):
        os.remove(os.path.join(os.getcwd(), 'gt.zip'))
    os.system(f'cd {parsed_gt_dir}; zip -q ../gt.zip *')
    return gt_boxes


def parse_gt_json(args):
    # removing unnecessary files and taking only the matching ones
    ims_list = sorted([file for file in os.listdir(args.test_img) if not file.startswith('.') and is_image(file)])
    img_list_prefix = [os.path.splitext(im)[0] for im in ims_list]
    gt_file_paths = glob.glob(f"{args.test_gt}/*.json")
    parsed_gt_dir = os.path.join(os.getcwd(), 'gt')
    if len(os.listdir(parsed_gt_dir)):
        os.system(f'rm {parsed_gt_dir}/*')
    os.makedirs(parsed_gt_dir, exist_ok=True)
    gt_boxes = []
    
    for gt_file in gt_file_paths:
        gt_prefix = os.path.splitext(os.path.basename(gt_file))[0]
        if gt_prefix in img_list_prefix:
            with open(gt_file) as fr:
                annotations = json.load(fr)
                for annotation in annotations['shapes']:
                    points_x = sorted(annotation['points'], key=lambda x: x[0])
                    points_y = sorted(annotation['points'], key=lambda y: y[1])
                    label = annotation['label']

                    # x1, y1 = points[0][0], points[0][1]
                    # x2, y2 = points[1][0], points[1][1]
                    # x3, y3 = points[2][0], points[2][1]
                    # x4, y4 = points[3][0], points[3][1]

                    xmin, ymin = points_x[0][0], points_y[0][1]
                    xmax, ymax = points_x[-1][0], points_y[-1][1]

                    gt_boxes.append(np.array([xmin, ymin, xmax, ymax], dtype=np.int32))
                    with open(os.path.join(parsed_gt_dir, f'{gt_prefix}.txt'), 'a+') as fa:
                        fa.write(f'{int(xmin)}, {int(ymin)}, {int(xmax)}, {int(ymax)}, ###\n')
    if os.path.exists(os.path.join(os.getcwd(), 'gt.zip')):
        os.remove(os.path.join(os.getcwd(), 'gt.zip'))
    os.system(f'cd {parsed_gt_dir}; zip -q ../gt.zip *')
    return gt_boxes


def evaluate(model, args):
    ims_list = sorted([file for file in os.listdir(args.test_img) if not file.startswith('.') and is_image(file)])
    img_list_prefix = [os.path.splitext(im)[0] for im in ims_list]
    gt_file_paths = glob.glob(f"{os.getcwd()}/gt/*.txt")
    parsed_pred_dir = os.path.join(os.getcwd(), 'pred')
    if len(os.listdir(parsed_pred_dir)):
        os.system(f'rm {parsed_pred_dir}/*')
    os.makedirs(parsed_pred_dir, exist_ok=True)
    
    total_iou = 0.
    total_scores = 0.
    count = 0
    pred_boxes = []
    
    for gt_file in gt_file_paths:
        gt_prefix = os.path.splitext(os.path.basename(gt_file))[0]
        if gt_prefix in img_list_prefix:
            im_path = os.path.join(args.test_img, ims_list[img_list_prefix.index(gt_prefix)])
            src = cv2.imread(im_path, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            h, w = im.shape[:2]

            cls_dets = im_detect(model, im, target_sizes=args.target_size)
            for j in range(len(cls_dets)):
                cls, scores = cls_dets[j, 0], cls_dets[j, 1]
                pred_box = cls_dets[j, 2:]
                pred_boxes.append([pred_box[0], pred_box[1], pred_box[2], pred_box[3]])
                total_scores += scores
                with open(os.path.join(parsed_pred_dir, f'{gt_prefix}.txt'), 'a+') as fa:
                    fa.write(f'{pred_box[0]}, {pred_box[1]}, {pred_box[2]}, {pred_box[3]}, ###\n')
    if os.path.exists(os.path.join(os.getcwd(), 'pred.zip')):
        os.remove(os.path.join(os.getcwd(), 'pred.zip'))
    os.system(f'cd {parsed_pred_dir}; zip -q ../pred.zip *')
    
    # precision, recall, f1, accuracy = calculate_precision_recall(tp, tn, fp, fn)
    json_schema = '{\"LTRB\":\"False\",\"E2E\":\"False\",\"GT_SAMPLE_NAME_2_ID\":\"([A-Za-z0-9-_() ]+).txt\",\"DET_SAMPLE_NAME_2_ID\":\"([A-Za-z0-9-_() ]+).txt\"}'
    result = master_evaluation('DetEva', os.path.join(os.getcwd(), 'gt.zip'), os.path.join(os.getcwd(), 'pred.zip'), json_schema)
    
    os.system(f'rm {os.getcwd()}/pred.zip')
    os.system(f'rm {os.getcwd()}/pred/*')
    return result
