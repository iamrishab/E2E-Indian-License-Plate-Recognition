import os
import cv2
import string
import yaml

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from tps.dataset import *
from tps.TPSNet import TPS


with open("config.yaml") as f:
    config = yaml.load(f)
    config = config['tps']


def infer(folder_path):
    tps_model = TPS(config)
    tps_model = torch.nn.DataParallel(tps_model).to(device)
    # tps_model.load_state_dict(torch.load(config['saved_model'], map_location=device))
    print('TPS model loaded')
    tps_model.eval()
    
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=config['imgH'], imgW=config['imgW'], keep_ratio_with_pad=config['PAD'])
    demo_data = RawDataset(root=folder_path, opt=config)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=config['batch_size'],
        shuffle=False,
        num_workers=int(config['workers']),
        collate_fn=AlignCollate_demo, pin_memory=True)
    
    with torch.no_grad():
        results, image_path_lists = [], []
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            result = tps_model(image, False)
            results.append(result)
            image_path_lists.append(image_path_list)
        return results, image_path_lists
    

if __name__ == '__main__':
    """ vocab / character number configuration """
    if config['sensitive']:
        config['character'] = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    config['num_gpu'] = torch.cuda.device_count()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    folder_path = '/home/rishab/licenseplaterecognition/LPRNet/dataset/train'
    save_folder = '/home/rishab/licenseplaterecognition/LPRNet/dataset/save' 
    os.makedirs(save_folder, exist_ok=True)
    
    results, image_path_lists = infer(folder_path)
    
    for tensors, image_paths in zip(results, image_path_lists):
        for tensor, image_path in zip(tensors, image_paths):
            image = tensor2im(tensor)
            save_path = os.path.join(save_folder, os.path.basename(image_path))
            cv2.imwrite(save_path, image)
            print(f'Saved image: {save_path}')