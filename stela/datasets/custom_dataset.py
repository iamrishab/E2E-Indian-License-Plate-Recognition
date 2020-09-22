import os
import cv2
import numpy as np
import torch
import torch.utils.data as data

from utils.bbox import quad_2_rbox


class CustomDataset(data.Dataset):
    """"""
    def __init__(self,
                 img_dir=None,
                 gt_dir=None,
                 dformat='txt'):
        self.image_ext = [".bmp", ".png", ".jpg", ".jpeg", ".JPG", ".JPEG"]
        self.gt_ext = [".txt", ".json"]
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.classes = ('__background__', 'text')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self.image_list = sorted([file for file in os.listdir(img_dir) if os.path.splitext(file)[-1] in self.image_ext])
        filenames = [os.path.splitext(file)[0] for file in self.image_list]
        self.annotation_list = sorted([file for file in os.listdir(gt_dir) if os.path.splitext(file)[0].split('gt_', 1)[-1] in filenames and os.path.splitext(file)[-1] in self.gt_ext])
        self.dformat = dformat

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        im_path = self._image_path_from_index(self.image_list[index])
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if im is not None:
            h, w = im.shape[:2]
        
        gt_path = self._annotation_path_from_index(self.annotation_list[index])
        if self.dformat == 'txt':
            roidb = self._load_annotation_txt(gt_path, h, w)
        elif self.dformat == 'json':
            roidb = self._load_annotation_json(gt_path)
        else:
            raise Exception('Data format not supported!')
        gt_inds = np.where(roidb['gt_classes'] != 0)[0]
        bboxes = roidb['boxes'][gt_inds, :]
        classes = roidb['gt_classes'][gt_inds]
        
        gt_boxes = np.empty((len(gt_inds), 6), dtype=np.float32)
        for i, bbox in enumerate(bboxes):
            gt_boxes[i, :5] = quad_2_rbox(np.array(bbox))
            gt_boxes[:, 5] = classes[i]
            
        return {'image': im, 'boxes': gt_boxes}
    
    
    def _load_annotation_txt(self, gt_path, h, w):
        # change the logic here for reading GT
        bboxes = []
        classes = []
        with open(gt_path, encoding="utf-8-sig", errors="surrogateescape") as fr:
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
                        # label = line[0]
                    except ValueError:
                        x1, y1, x2, y2 = map(float, line[0:4])
                        # label = ''.join(line[4:])
                    except Exception:
                        # print(line)
                        continue
                    if x1 < x2 <= 1. and y1 < y2 <= 1.:
                        x1, y1, x2, y2 = x1*w, y1*h, x2*w, y2*h
                    x1, y1, x2, y2, x3, y3, x4, y4 = x1, y1, x2, y1, x2, y2, x1, y2
                
                elif len(line) == 9:
                    try:
                        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line[-8:])
                        # label = line[0]
                    except ValueError:
                        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line[0:8])
                        # label = ''.join(line[8:])
                    except Exception:
                        print(line)
                        continue
                    if x1 < x3 <= 1. and y1 < y3 <= 1.:
                        x1, y1, x2, y2, x3, y3, x4, y4 = x1*w, y1*h, x2*w, y2*h, x3*w, y3*h, x4*w, y4*h
                
                classes.append(self.class_to_ind['text'])
                bboxes.append(np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.int32))
        
        return {'boxes': np.array(bboxes, dtype=np.int32), 'gt_classes': np.array(classes)}
    
    
    def _load_annotation_json(self, gt_path):
        # change the logic here for reading GT
        bboxes = []
        classes = []
        with open(gt_path) as fr:
            annotations = json.load(fr)
            for annotation in annotations['shapes']:
                points_x = sorted(annotation['points'], key=lambda x: x[0])
                points_y = sorted(annotation['points'], key=lambda y: y[1])
                label = annotation['label']
                
                # x1, y1 = points[0][0], points[0][1]
                # x2, y2 = points[1][0], points[1][1]
                # x3, y3 = points[2][0], points[2][1]
                # x4, y4 = points[3][0], points[3][1]
                
                xmin, ymin = int(points_x[0][0]), int(points_y[0][1])
                xmax, ymax = int(points_x[-1][0]), int(points_y[-1][1])
                classes.append(self.class_to_ind['text'])
                
                # bboxes.append(np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.int32))
                bboxes.append(np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax], dtype=np.int32))
        
        return {'boxes': np.array(bboxes, dtype=np.int32), 'gt_classes': np.array(classes)}


    def _image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self.img_dir, index)
        if os.path.exists(image_path):
            return image_path
        
        if not image_exist:
            raise Exception('Image path does not exist: {}'.format(
                os.path.join(self.img_dir, index))
            )
        
    def _annotation_path_from_index(self, index):
        gt_path = os.path.join(self.gt_dir, index)
        if os.path.exists(gt_path):
            return gt_path
        
        if not gt_path:
            raise Exception('Annotation path does not exist: {}'.format(
                os.path.join(self.gt_dir, index))
            )