# -*- coding: utf-8 -*-
# /usr/bin/env/python3

import time
import os
import shutil
import yaml
import string
import yaml
import cv2
import numpy as np
import datetime

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from lprnet.load_data import CHARS, CHARS_DICT, LPRDataLoader
from tps.utils import *
from tps.dataset import *
from model import LPR_TPS
sys.path.append('/home/rishab')
from licenseplaterecognition.utils.mlflow import MLflowTrackingRestApi

# for reproducability of results
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

log_dir = f'runs/lpr_tps_experiment_{str(datetime.datetime.utcnow())}'
writer = SummaryWriter(log_dir)
print(f'Writing logs to {log_dir}')
mlflow_rest = MLflowTrackingRestApi('127.0.0.1', '5000', '1')

mlflow_rest.set_tag({'key': "mlflow.runName", 'value': 'lpr-reco-v0'})
mlflow_rest.set_tag({'key': "mlflow.user", 'value': 'rishab'})
mlflow_rest.set_tag({'key': "backbone", 'value': 'TPS-LPRNet'})

mlflow_rest.log_param({'key': "platform", 'value': 'pytorch'})
mlflow_rest.log_param({'key': "platform_version", 'value': str(torch.__version__)})
# mlflow_rest.log_param({'key': "training_steps", 'value': str(args.training_steps)})
# mlflow_rest.log_param({'key': "training_set", 'value': str(args.bucket_url)})
# mlflow_rest.log_param({'key': "validation_set", 'value': str(args.bucket_url)})


# helper functions
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


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


def train():
    T_length = config['lpr']['lpr_max_len']
    epoch = 0 + config['lpr']['resume_epoch']
    loss_val = 0

    if not os.path.exists(config['lpr']['save_folder']):
        os.mkdir(config['lpr']['save_folder'])

    model = LPR_TPS(config)
    
    device = torch.device("cuda:0" if config['lpr']['cuda'] else "cpu")
    print('Running on device:', device)
    model.to(device)
    print("Successful to build network!")

    # load pretrained model
    if config['internal']['pretrained_model']:
        model.load_state_dict(torch.load(config['internal']['pretrained_model']))
        print("load pretrained model successful!")
    else:
        # weight initialization for LPRNet
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

        model.lpr.backbone.apply(weights_init)
        model.lpr.container.apply(weights_init)
        
        # weight initialization for TPS
        for name, param in model.transformation.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        
        print("initial net weights successful!")

    # define optimizer
    if config['lpr']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    elif config['lpr']['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lpr']['learning_rate'],
                              momentum=config['lpr']['momentum'], weight_decay=config['lpr']['weight_decay'])
    elif config['lpr']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['lpr']['learning_rate'], alpha = 0.9, eps=1e-08,
                             momentum=config['lpr']['momentum'], weight_decay=config['lpr']['weight_decay'])
    else:
        raise Exception('No optimizer specified')

    # dataset
    train_img_dirs = os.path.expanduser(config['lpr']['train_img_dirs'])
    test_img_dirs = os.path.expanduser(config['lpr']['test_img_dirs'])
    print(f'Train image dir: {train_img_dirs}, samples: {len(os.listdir(train_img_dirs))}', )
    print(f'Test image dir: {test_img_dirs}, samples: {len(os.listdir(test_img_dirs))}', )
    
    # cheking for improper files in dir
    # [shutil.rmtree(os.path.join(train_img_dirs, file)) for file in os.listdir(train_img_dirs) if file.startswith('.')]
    # [shutil.rmtree(os.path.join(test_img_dirs, file)) for file in os.listdir(test_img_dirs) if file.startswith('.')]
    
    [os.system(f"rm -rf {os.path.join(train_img_dirs, file)}") for file in os.listdir(train_img_dirs) if file.startswith('.')]
    [os.system(f"rm -rf {os.path.join(test_img_dirs, file)}") for file in os.listdir(test_img_dirs) if file.startswith('.')]
    
    # specifying training specs
    epoch_size = len(os.listdir(train_img_dirs)) // config['lpr']['train_batch_size']
    print('epoch_size:', epoch_size)
    max_iter = config['lpr']['max_epoch'] * epoch_size

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    if config['lpr']['resume_epoch'] > 0:
        start_iter = config['lpr']['resume_epoch'] * epoch_size
    else:
        start_iter = 0
    
    accuracy = 0.
    best_accuracy = 0.
    
    AlignCollate_train = AlignCollate(imgH=config['tps']['imgH'], imgW=config['tps']['imgW'], keep_ratio_with_pad=config['tps']['PAD'])
    train_data = RawDataset(root=train_img_dirs, opt=config)  # use RawDataset
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config['tps']['batch_size'],
                                    shuffle=False,
                                    num_workers=int(config['tps']['workers']),
                                    collate_fn=AlignCollate_train, pin_memory=True)

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(train_dataloader)
            loss_val = 0
            epoch += 1

        if (iteration + 1) % config['lpr']['test_interval'] == 0:
            accuracy = Greedy_Decode_Eval(model, test_img_dirs, config)
            mlflow_rest.log_metric(metric = {'key': 'accuracy', 'value': accuracy, 'step': iteration})
            model.train() # should be switch to train mode
            
        if iteration !=0 and accuracy > best_accuracy: # iteration % config['save_interval'] == 0:
            torch.save(model.state_dict(), config['lpr']['save_folder'] + 'LPRNet_TPS_' + 'acc_' + f'{round(best_accuracy, 3)}' + '_iteration_' + repr(iteration) + '.pth')
            best_accuracy = accuracy

        start_time = time.time()
        # load train data
        images, paths, labels, lengths = next(batch_iterator)
        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, config['lpr']['learning_rate'], config['lpr']['lr_schedule'])

        if config['lpr']['cuda']:
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # forward
        logits = model(images)
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
            print('Epoch:' + repr(epoch) + ' || iter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item()) +
                  'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))
            mlflow_rest.log_metric(metric = {'key': 'loss', 'value': loss.item(), 'step': iteration})
            # log the running loss
            writer.add_scalar('training loss',
                            loss.item(),
                            epoch * len(batch_iterator) + iteration)
            
#             writer.add_figure('predictions vs. actuals',
#                             plot_classes_preds(model, images, labels),
#                             global_step=epoch * len(batch_iterator) + iteration)
    # final test
    print("Final test Accuracy:")
    accuracy = Greedy_Decode_Eval(model, test_img_dirs, config)
    # save final parameters
    torch.save(model.state_dict(), config['lpr']['save_folder'] + 'Final_LPRNet_TPS_model.pth')


if __name__ == "__main__":
    train()
