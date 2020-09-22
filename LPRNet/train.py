# -*- coding: utf-8 -*-
# /usr/bin/env/python3

import sys
import numpy as np
import time
import os
import shutil
import yaml
# import mlflow

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn

from lprnet.load_data import CHARS, CHARS_DICT, LPRDataLoader
from lprnet.LPRNet import LPRNet
from lprnet.utils import *
sys.path.append('/home/rishab')
from licenseplaterecognition.utils.mlflow import MLflowTrackingRestApi

# for reproducability of results
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

mlflow_rest = MLflowTrackingRestApi('127.0.0.1', '5000', '1')


with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = config['lpr']


def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class GradientClipping:
    def __init__(self, model, clip=0.):
        self.model, self.clip = model, clip
    def on_backward_end(self, **kwargs):
        if self.clip:
            nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
            
            
class LearningRateScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def on_batch_begin(self, iteration, **kwargs):
        # control the learning rate over iteration
        self.optimizer.lr = fct(iteration)


def train():
    T_length = config['lpr_max_len']
    epoch = 0 + config['resume_epoch']
    loss_val = 0

    if not os.path.exists(config['save_folder']):
        os.mkdir(config['save_folder'])
    lprnet = LPRNet(lpr_max_len=config['lpr_max_len'], phase=config['phase_train'], class_num=len(CHARS), dropout_rate=config['dropout_rate'])
    lprnet.train()
    device = torch.device("cuda:0" if config['cuda'] else "cpu")
    print('Running on device:', device)
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if config['pretrained_model']:
        lprnet.load_state_dict(torch.load(config['pretrained_model']))
        print("load pretrained model successful!")
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")

    # define optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(lprnet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(lprnet.parameters(), lr=config['learning_rate'],
                              momentum=config['momentum'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(lprnet.parameters(), lr=config['learning_rate'], alpha = 0.9, eps=1e-08,
                             momentum=config['momentum'], weight_decay=config['weight_decay'])
    else:
        raise Exception('No optimizer specified')

    # dataset
    train_img_dirs = os.path.expanduser(config['train_img_dirs'])
    test_img_dirs = os.path.expanduser(config['test_img_dirs'])
    print(f'Train image dir: {train_img_dirs}, samples: {len(os.listdir(train_img_dirs))}', )
    print(f'Test image dir: {test_img_dirs}, samples: {len(os.listdir(test_img_dirs))}', )
    
    [os.system(f"rm -rf {os.path.join(train_img_dirs, file)}") for file in os.listdir(train_img_dirs) if file.startswith('.')]
    [os.system(f"rm -rf {os.path.join(test_img_dirs, file)}") for file in os.listdir(test_img_dirs) if file.startswith('.')]
    
    train_dataset = LPRDataLoader(train_img_dirs, config['img_size'], config['lpr_max_len'])
    
    # specifying training specs
    epoch_size = len(os.listdir(train_img_dirs)) // config['train_batch_size']
    print('epoch_size:', epoch_size)
    max_iter = config['max_epoch'] * epoch_size

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    if config['resume_epoch'] > 0:
        start_iter = config['resume_epoch'] * epoch_size
    else:
        start_iter = 0
    
    accuracy = 0.
    best_accuracy = 0.

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(DataLoader(train_dataset, config['train_batch_size'], shuffle=True, num_workers=config['num_workers'], collate_fn=collate_fn))
            loss_val = 0
            epoch += 1

        if (iteration + 1) % config['test_interval'] == 0:
            accuracy = Greedy_Decode_Eval(lprnet, test_img_dirs, config)
            mlflow_rest.log_metric(metric = {'key': 'accuracy', 'value': accuracy, 'step': iteration})
            lprnet.train() # should be switch to train mode
            
        if iteration !=0 and accuracy > best_accuracy: # iteration % config['save_interval'] == 0:
            torch.save(lprnet.state_dict(), config['save_folder'] + 'LPRNet_' + 'acc_' + f'{round(best_accuracy, 3)}' + '_iteration_' + repr(iteration) + '.pth')
            best_accuracy = accuracy

        start_time = time.time()
        # load train data
        images, labels, lengths = next(batch_iterator)
        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, config['learning_rate'], config['lr_schedule'])

        if config['cuda']:
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # forward
        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        log_probs = log_probs.log_softmax(2).requires_grad_()
        
        # backprop
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        if loss.item() == np.inf:
            continue
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
        end_time = time.time()
        
        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epoch-iter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item()) +
                  'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))
            mlflow_rest.log_metric(metric = {'key': 'loss', 'value': loss.item(), 'step': iteration})
            
    # final test
    print("Final test Accuracy:")
    accuracy = Greedy_Decode_Eval(lprnet, test_img_dirs, config)
    # save final parameters
    torch.save(lprnet.state_dict(), config['save_folder'] + 'Final_LPRNet_model.pth')


if __name__ == "__main__":
    train()
