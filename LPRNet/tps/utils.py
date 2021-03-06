import os
import cv2
import time
import numpy as np
from fuzzywuzzy import process
from Levenshtein import ratio as lev_ratio

import torch
from torch.utils.data import *
from torch.autograd import Variable

from .dataset import CHARS, CHARS_DICT, AlignCollate, RawDataset


def Greedy_Decode_Eval(lprnet, datasets, config):
    lprnet.eval()
    epoch_size = len(os.listdir(datasets)) // config['tps']['batch_size']

    AlignCollate_test = AlignCollate(imgH=config['tps']['imgH'], imgW=config['tps']['imgW'], keep_ratio_with_pad=config['tps']['PAD'])
    test_data = RawDataset(root=datasets, opt=config)  # use RawDataset
    
    # create batch iterator
    test_dataloader = DataLoader(
                            test_data, batch_size=config['tps']['batch_size'],
                            shuffle=False,
                            num_workers=int(config['tps']['workers']),
                            collate_fn=AlignCollate_test, pin_memory=True)

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        if i % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(test_dataloader)
        
        # load train data
        images, _, labels, lengths = next(batch_iterator)
        
        start = 0
        targets = []
        
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        targets = np.array([el.numpy() for el in targets])

        if config['lpr']['cuda']:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = lprnet(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeat label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
            
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                actual = "".join([CHARS[i] for i in np.asarray(targets[i])])
                predicted = "".join([CHARS[i] for i in np.asarray(label)])
                # print(f'A: {actual} P: {predicted} D: {lev_ratio(actual, predicted)}')
                Tn_1 += 1
            elif (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                actual = "".join([CHARS[i] for i in np.asarray(targets[i])])
                predicted = "".join([CHARS[i] for i in np.asarray(label)])
                # print(f'A: {actual} P: {predicted} D: {lev_ratio(actual, predicted)}')
                Tn_2 += 1
    
    delta = 1e-3
    Acc = Tp / (Tp + Tn_1 + Tn_2 + delta)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s".format((t2 - t1) / len(os.listdir(datasets))))
    return Acc