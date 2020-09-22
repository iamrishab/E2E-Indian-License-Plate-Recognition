from __future__ import print_function

import os
import sys
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

sys.path.append('/home/rishab')

# from datasets.voc_dataset import VOCDataset
from datasets.custom_dataset import CustomDataset
from datasets.collater import Collater
from models.stela import STELA
from utils.timer import Timer
from custom_eval import *
from licenseplaterecognition.utils.mlflow import MLflowTrackingRestApi


_t = Timer()
mlflow_rest = MLflowTrackingRestApi('127.0.0.1', '5000', '0')


def train_model(args):
    # train
    train_dataset = CustomDataset(args.train_img, args.train_gt, args.gt_type_train)
    print('Number of Training Images is: {}'.format(len(train_dataset)))
    scales = args.training_size + 32 * np.array([x for x in range(-5, 6)])
    collater = Collater(scales=scales, keep_ratio=False, multiple=32)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=collater,
        shuffle=True,
        drop_last=True
    )
    
    os.makedirs('./weights', exist_ok=True)
    
    if args.gt_type_test == 'json':
        parse_gt_json(args)
    elif args.gt_type_test == 'txt':
        parse_gt_txt(args)
        
    model = STELA(backbone=args.backbone, num_classes=2)
    if os.path.exists(args.pretrained):
        model.load_state_dict(torch.load(args.pretrained))
        print('Load pretrained model from {}.'.format(args.pretrained))
    if torch.cuda.is_available():
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    iters_per_epoch = np.floor((len(train_dataset) / float(args.batch_size)))
    num_epochs = int(np.ceil(args.max_iter / iters_per_epoch))
    iter_idx = 0
    best_loss = sys.maxsize
    for _ in range(num_epochs):
        for _, batch in enumerate(train_loader):
            iter_idx += 1
            if iter_idx > args.max_iter:
                break
            _t.tic()
            model.train()
            
            if args.freeze_bn:
                if torch.cuda.device_count() > 1:
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()

            optimizer.zero_grad()
            ims, gt_boxes = batch['image'], batch['boxes']
            if torch.cuda.is_available():
                ims, gt_boxes = ims.cuda(), gt_boxes.cuda()
            losses = model(ims, gt_boxes)
            loss_cls, loss_reg = losses['loss_cls'].mean(), losses['loss_reg'].mean()
            if losses.__contains__('loss_ref'):
                loss_ref = losses['loss_ref'].mean()
                loss = loss_cls + (loss_reg + loss_ref) * 0.5
            else:
                loss = loss_cls + loss_reg
            if bool(loss == 0):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            scheduler.step()
            
            if iter_idx % args.display == 0:
                info = 'iter: [{}/{}], time: {:1.3f}'.format(iter_idx, args.max_iter, _t.toc())
                if losses.__contains__('loss_ref'):
                    info = info + ', ref: {:1.3f}'.format(loss_ref.item())
                    mlflow_rest.log_metric(metric = {'key': 'loss_ref', 'value': loss_ref.item(), 'step': iter_idx})
                print(info + ', loss_cls: {:1.3f}, loss_reg: {:1.3f}, total_loss: {:1.3f}'.format(loss_cls.item(), loss_reg.item(), loss.item()))                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), 'weights/weight_{}_{:1.3f}.pth'.format(iter_idx, loss.item()))
                    else:
                        torch.save(model.state_dict(), 'weights/weight_{}_{:1.3f}.pth'.format(iter_idx, loss.item()))
                mlflow_rest.log_metric(metric = {'key': 'loss_cls', 'value': loss_cls.item(), 'step': iter_idx})
                mlflow_rest.log_metric(metric = {'key': 'loss_reg', 'value': loss_reg.item(), 'step': iter_idx})
                mlflow_rest.log_metric(metric = {'key': 'total_loss', 'value': loss.item(), 'step': iter_idx})
            
#             if (arg.eval_iter > 0) and (iter_idx % arg.eval_iter) == 0:
                
                ## mlflow_rest.log_metric(metric = {'key': 'accuracy', 'value': accuracy, 'step': iter_idx})
                ## mlflow_rest.log_metric(metric = {'key': 'IOU', 'value': avg_iou, 'step': iter_idx})
                ## mlflow_rest.log_metric(metric = {'key': 'confidence', 'value': confidence, 'step': iter_idx})
                ## print('IOU: {}, Score: {}'.format(avg_iou, confidence))
                ## print(f"precision: {precision*100}, recall: {recall*100}, f1: {f1*100}, accuracy: {accuracy*100}")
                
            if iter_idx % args.save_interval == 0:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), f'weights/check_{iter_idx}.pth')
                else:
                    torch.save(model.state_dict(), f'weights/check_{iter_idx}.pth')
    
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), f'weights/final_{args.max_iter}.pth')
    else:
        torch.save(model.state_dict(), f'weights/final_{args.max_iter}.pth')

    model.eval()
    if torch.cuda.device_count() > 1:
        result = evaluate(model.module, args)
    else:
        result = evaluate(model, args)

    mlflow_rest.log_metric(metric = {'key': 'precision', 'value': result['precision'], 'step': iter_idx})
    mlflow_rest.log_metric(metric = {'key': 'recall', 'value': result['recall'], 'step': iter_idx})
    mlflow_rest.log_metric(metric = {'key': 'hmean', 'value': result['hmean'], 'step': iter_idx})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    # network
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--freeze_bn', type=bool, default=False)
    parser.add_argument('--pretrained', type=str, default='weights/final_30000.pth')
    # dataset
#     parser.add_argument('--train_img', type=str, default='/home/rishab/icdar2015/combined_img')
#     parser.add_argument('--train_gt', type=str, default='/home/rishab/icdar2015/combined_gt')
    parser.add_argument('--train_img', type=str, default='/home/rishab/lpr_dataset/Data-Images/Cars')
    parser.add_argument('--train_gt', type=str, default='/home/rishab/lpr_dataset/Data-Images/Labels')
    parser.add_argument('--gt_type_train', type=str, default='txt')
    # training
    parser.add_argument('--training_size', type=int, default=640)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument('--step_size', type=int, default=20000)
    parser.add_argument('--display', type=int, default=100)
    parser.add_argument('--save_interval', type=str, default=10000)
    # testing
    parser.add_argument('--eval_iter', type=int, default=1000)
    parser.add_argument('--target_size', type=int, default=[800])
    parser.add_argument('--test_img', type=str, default='/home/rishab/lpr_dataset/labelled/13 -May-2020/Car Images/JSON')
    parser.add_argument('--test_gt', type=str, default='/home/rishab/lpr_dataset/labelled/13 -May-2020/Car Images/JSON')
    parser.add_argument('--eval_dir', type=str, default='./eval/')
    parser.add_argument('--gt_type_test', type=str, default='json')
    #
    arg = parser.parse_args()
    train_model(arg)
