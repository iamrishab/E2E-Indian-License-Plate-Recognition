# -*- coding: utf-8 -*-
# /usr/bin/env/python3

import time
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

import torch
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn

from lprnet.load_data import CHARS, CHARS_DICT, LPRDataLoader
from lprnet.utils import *
from tps.dataset import *
from model import LPR_TPS


with open("config.yaml") as f:
    config = yaml.load(f)


def test():
    lprnet = LPR_TPS(config)
    lprnet.eval()
    device = torch.device("cuda:0" if config['lpr']['cuda'] else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if config['lpr']['pretrained_model']:
        lprnet.load_state_dict(torch.load(config['lpr']['pretrained_model']))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False
    
    test_img_dirs = os.path.expanduser(config['test_img_dirs'])
    
    [os.system(f"rm -rf {os.path.join(test_img_dirs, file)}") for file in os.listdir(test_img_dirs) if file.startswith('.')]
    
    
    AlignCollate_train = AlignCollate_v2(imgH=config['tps']['imgH'], imgW=config['tps']['imgW'], keep_ratio_with_pad=config['tps']['PAD'])
    test_dataset = RawDataset_v2(root=train_img_dirs, opt=config)  # use RawDataset
    
    accuracy = Greedy_Decode_Eval(lprnet, config['test_img_dirs'], config)
    

if __name__ == "__main__":
    test()
