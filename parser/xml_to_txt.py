import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET


# change here for your own mappings
class_to_ind = {
    'nplate': 0
}


def convert_pascal_annotation_to_txt(img_path, gt_path, save_path):
    """
    Load image and bounding boxes info from XML file
    """
    os.makedirs(save_path, exist_ok=True)
    [os.system(f"rm -rf {os.path.join(img_path, file)}") for file in os.listdir(img_path) if file.startswith('.')]
    [os.system(f"rm -rf {os.path.join(gt_path, file)}") for file in os.listdir(gt_path) if file.startswith('.')]
    imgs = sorted(os.listdir(img_path))
    gts = sorted(os.listdir(gt_path))
    print('Total images:', len(imgs))
    print('Total gts:', len(gts))
    
    for img, gt in zip(imgs, gts):
        print('Processing:', img, gt)
        name, _ = os.path.splitext(img)
        gt_file = os.path.join(gt_path, gt)
        tree = ET.parse(gt_file)
        objs = tree.findall('object')
        h, w = cv2.imread(os.path.join(img_path, img)).shape[:2]
        boxes, gt_classes = [], []
        for _, obj in enumerate(objs):
            bnd_box = obj.find('bndbox')
            box = [
                float(bnd_box.find('xmin').text)/w,
                float(bnd_box.find('ymin').text)/h,
                float(bnd_box.find('xmax').text)/w,
                float(bnd_box.find('ymax').text)/h,
            ]
            label = class_to_ind[obj.find('name').text.lower().strip()]
            boxes.append(box)
            gt_classes.append(label)
        
        with open(os.path.join(save_path, name+'.txt'), 'w') as fw:
            for box, gt in zip(boxes, gt_classes):
                x1, y1, x2, y2 = box
                fw.write('{} {} {} {} {} {} {} {} {}\n'.format(gt, x1, y1, x2, y1, x2, y2, x1, y2))


if __name__ == '__main__':
    img_path = '/home/rishab/lpr_dataset/collect/images'
    gt_path = '/home/rishab/lpr_dataset/collect/annotations'
    save_path = '/home/rishab/lpr_dataset/collect/annotations_xml'
    convert_pascal_annotation_to_txt(img_path, gt_path, save_path)
