# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from datasets.make_dataloader import make_dataset
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from tool.utils_server import save_network
from tool.utils_server import load_network
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from datasets.queryDataset import Dataset_query,Query_transforms
from models.AEN.model import AEN
from torch.optim import lr_scheduler


#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='AEN', type=str, help='output model name')
parser.add_argument('--data_dir', default='E:\\University-Release\\train', type=str, help='training dir path')
parser.add_argument('--checkpoint', default='.pth', type=str, help='one stage model path')
parser.add_argument('--batchsize', default=7, type=int, help='batchsize') # batch is the number of T
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--num_worker',default=0, type=int,help='')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation')
parser.add_argument('--pad', default=0, type=int, help='')
parser.add_argument('--epoch', default=120, type=int, help='epoch')
parser.add_argument('--block', default=2, type=int, help='num of part')
parser.add_argument('--nclasses', default=701, type=int, help='')
parser.add_argument('--outdim', default=2048, type=int, help='outdim of AEN')

opt = parser.parse_args()

opt.views = 2
opt.share = True
opt.sample_num = opt.batchsize    # num of repeat sampling

d_model = (opt.block + 1) * 512
model_2 = AEN(d_model=d_model, outdim=opt.outdim)


model = load_network(opt)
with torch.no_grad():
    model.eval()
model = model.cuda()
dataloaders,class_names,dataset_sizes = make_dataset(opt)


optim = torch.optim.SGD(model_2.parameters(),lr=0.01,weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,milestones=[20,110],gamma=0.1)
loss_funtion = nn.MSELoss()

for i in range(opt.epoch):
    running_loss = 0.0
    for data_s,data_d in dataloaders:
        inputs_s, labels_s = data_s
        inputs_d, labels_d = data_d
        input_img_s = Variable(inputs_s.cuda())
        input_img_d = Variable(inputs_d.cuda())
        outputs_s, outputs_d = model(input_img_s, input_img_d)
        outputs_s = outputs_s.view(outputs_s.shape[0], -1, 1).squeeze(-1)
        outputs_d = outputs_d.view(outputs_d.shape[0], -1, 1).squeeze(-1)
        optim.zero_grad()

        outputs_1, outputs_2 = model_2(queries=outputs_s, support_set=outputs_d)
        outputs_1, outputs_2 = outputs_1.squeeze(), outputs_2.squeeze()
        loss = loss_funtion(outputs_1, outputs_2)
        loss.backward()
        optim.step()
        running_loss = running_loss + loss
    print("epoch:", i, "loss:", running_loss)
    # if i>=60:
    save_network(model_2, opt.name, i)

