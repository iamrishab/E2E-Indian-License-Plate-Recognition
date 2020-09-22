import os
import string
import argparse
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import *
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time 

from flask import Flask, Response, request, redirect, jsonify
import json


app = Flask(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
parser.add_argument('--saved_model', default='saved_models/best_accuracy.pth', help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', type=bool, default=False, help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', help='character label')
parser.add_argument('--sensitive', type=bool, default=True, help='for sensitive character mode')
parser.add_argument('--PAD', type=bool, default=False, help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, default='None', help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

opt = parser.parse_args()

""" vocab / character number configuration """
if opt.sensitive:
    opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()


""" model configuration """
if 'CTC' in opt.Prediction:
    converter = CTCLabelConverter(opt.character)
else:
    converter = AttnLabelConverter(opt.character)
opt.num_class = len(converter.character)

if opt.rgb:
    opt.input_channel = 3
model = Model(opt)
model = torch.nn.DataParallel(model).to(device)

# load model
# print('loading pretrained model from %s' % opt.saved_model)
model.load_state_dict(torch.load(opt.saved_model, map_location=device))
# predict
model.eval()


def infer(img_path):    
    if opt.rgb:
        img = Image.open(img_path).convert('RGB')  # for color image
    else:
        img = Image.open(img_path).convert('L')
    
    if opt.PAD:  # same concept with 'Rosetta' paper
        resized_max_w = opt.imgW
        input_channel = 3 if img[0].mode == 'RGB' else 1
        transform = NormalizePAD((input_channel, opt.imgH, resized_max_w))

        w, h = image.size
        ratio = w / float(h)
        if math.ceil(opt.imgH * ratio) > opt.imgW:
            resized_w = opt.imgW
        else:
            resized_w = math.ceil(opt.imgH * ratio)

        resized_image = image.resize((resized_w, opt.imgH), Image.BICUBIC)
        resized_image = transform(resized_image)
        image_tensor = torch.cat([resized_image.unsqueeze(0)], 0)
    else:
        transform = ResizeNormalize((opt.imgW, opt.imgH))
        image_tensor = transform(img)
        image_tensor = torch.cat([image_tensor.unsqueeze(0)], 0)
    
    texts = []
    with torch.no_grad():
        batch_size = image_tensor.size(0)
        image = image_tensor.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred).log_softmax(2)
            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        else:
            preds = model(image, text_for_pred, is_train=False)
            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
        
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]
            # calculate confidence score (= multiply of pred_max_prob)
            if pred != '':
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                # texts.append([pred, confidence_score.cpu().numpy().tolist()])
                texts.append(pred)
            else:
                texts.append('Undetected')
                # texts.append([pred, 0.])  
    return texts


@app.route("/deeptext",methods=["GET", "POST"])
def main():
    folder_path = request.form['path']
    texts = infer(folder_path)
    return json.dumps(texts)

            
if __name__ == '__main__':
    app.run('0.0.0.0', '8002', debug=False, threaded=True)

