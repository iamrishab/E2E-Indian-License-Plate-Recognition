import os
import cv2
import glob
import json
import numpy as np

from imutils import paths
from torch.utils.data import *


CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
         'U', 'V', 'W', 'X', 'Y', 'Z',
         ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

SUPPORTED_IMG_FORMAT = ['.png', '.jpg', '.jpeg']


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len):
        self.img_dir = img_dir
        self.img_paths = sorted([os.path.join(img_dir, file) for file in os.listdir(img_dir) if os.path.splitext(file)[-1].lower() in SUPPORTED_IMG_FORMAT])
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_file = self.img_paths[index]
        Image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        height, width = Image.shape[:2]
        if len(Image.shape) == 2:
            Image = np.expand_dims(Image, axis=2)
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.transform(Image)

        label, suffix = os.path.splitext(os.path.basename(img_file))
        label = [CHARS_DICT[c] for c in self.normalize_text(label)]
        if len(label) > self.lpr_max_len:
            raise Exception(f'Image: {imgname} | Length of label{len(label)} exceed max label size : {self.lpr_max_len}')
        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img
    
    def normalize_text(self, txt):
        # replace special characters
        for x in list(txt):
            if x not in CHARS:
                txt = txt.replace(x, '') 
        return txt
        