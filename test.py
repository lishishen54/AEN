import torch.nn as nn
import torch
import numpy as np
import time
import math
from torch.autograd import Variable
from datasets.queryDataset import Dataset_query,Query_transforms
import argparse
from torchvision import datasets, models, transforms
import os
from tqdm import tqdm
from tool.utils_server import save_network
from tool.utils_server import load_network
from models.AEN.model import AEN
import scipy.io

# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='AEN', type=str, help='name')
# parser.add_argument('--test_dir',default='E:\SUES-ALL\Testing\\300',type=str, help='./test_data')
parser.add_argument('--test_dir',default='E:\\University-Release\\test',type=str, help='./test_data')
parser.add_argument('--checkpoint', default='85.12 87.21 89.30 84.17 ELG.pth', type=str, help='save model path')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--checkpoint_CA', default='003', type=str, help='CA name')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--test_mode',default='2', type=int,help='1:satellite->drone    2:drone->satellite')
parser.add_argument('--num_worker',default=2, type=int,help='1:drone->satellite   2:satellite->drone')
parser.add_argument('--pad', default=0, type=int, help='')

opt = parser.parse_args()


opt.views = 2
opt.nclasses = 701
opt.block = 2
test_dir = opt.test_dir
data_dir = test_dir

model = load_network(opt)
with torch.no_grad():
    model.eval()
model = model.cuda()

model_2 = AEN(d_model=512 * (opt.block + 1), outdim=2048)
model_2.load_state_dict(torch.load('checkpoints/'+ opt.name + '/net_' + opt.checkpoint_CA + '.pth'))

with torch.no_grad():
    model_2.eval()
model_2 = model_2.cuda()



data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_query_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        Query_transforms(pad=opt.pad,size=opt.w),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


image_datasets_query = {x: datasets.ImageFolder(os.path.join(data_dir,x) ,data_query_transforms) for x in ['query_satellite','query_drone']}

image_datasets_gallery = {x: datasets.ImageFolder(os.path.join(data_dir,x) ,data_transforms) for x in ['gallery_satellite','gallery_drone']}

image_datasets = {**image_datasets_query, **image_datasets_gallery}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                         shuffle=False, num_workers=opt.num_worker) for x in ['gallery_satellite', 'gallery_drone','query_satellite','query_drone']}
# print(dataloaders['query_drone'])

ms = [1]

if opt.test_mode==1:
    query_name = 'query_satellite'
    gallery_name = 'gallery_drone'
elif opt.test_mode==2:
    query_name = 'query_drone'
    gallery_name = 'gallery_satellite'
else:
    raise Exception("opt.mode is not required")



def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1

since = time.time()
view_index = 1

def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

which_gallery = which_view(gallery_name)
which_query = which_view(query_name)

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,model_2,dataloaders, view_index = 1,d_model=512):
    features = torch.FloatTensor()
    for data in tqdm(dataloaders):
        img, label = data
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear',
                                                      align_corners=False)
                if view_index == 1:
                    outputs, _ = model(input_img, None)


                    #2维
                    outputs = outputs.view(outputs.shape[0], -1, 1).squeeze(-1)

                    meaningless = torch.ones_like(outputs)
                    outputs_2, _ = model_2(queries=outputs, support_set=meaningless)


                elif view_index == 3:
                     _, outputs = model(None, input_img)
                     outputs = outputs.view(outputs.shape[0], -1, 1).squeeze(-1)
                     meaningless = torch.ones_like(outputs)
                     _, outputs_2 = model_2(queries=meaningless,support_set=outputs)



                if i == 0:
                    # ff = outputs
                    # # temp = ff[ff.size(0),-1,1].squeeze(-1)
                    # temp = ff[:,:1536]
                    zz = outputs_2
                    # input = [zz, temp]
                    result = zz
                    # result = outputs_2
                else:
                    # ff += outputs  #三维，[batch_size, d_model, block+1]
                    # # temp = ff[ff.size(0),-1,1].squeeze(-1)
                    # temp = ff[:,:1536]
                    zz += outputs_2
                    # input = [zz, temp]
                    result = zz
                    # result += outputs_2

        # norm feature
        if len(result.shape) == 3:
            fnorm = torch.norm(result, p=2, dim=1, keepdim=True) * np.sqrt(ff.size(-1))
            result = result.div(fnorm.expand_as(result))
            result = result.view(result.size(0), -1)
        else:
            fnorm = torch.norm(result, p=2, dim=1, keepdim=True)
            result = result.div(fnorm.expand_as(result))

        features = torch.cat((features, result.data.cpu()), 0)  #二维，[batch_size, d_model*(block+1)]
    return features

gallery_path = image_datasets[gallery_name].imgs
f = open('gallery_name.txt','w')
for p in gallery_path:
    f.write(p[0]+'\n')
query_path = image_datasets[query_name].imgs
f = open('query_name.txt','w')
for p in query_path:
    f.write(p[0]+'\n')

gallery_label, gallery_path  = get_id(gallery_path)
query_label, query_path  = get_id(query_path)


if __name__ == "__main__":
    print("CA name = " + opt.checkpoint_CA)
    with torch.no_grad():
        query_feature = extract_feature(model,model_2,dataloaders[query_name], which_query)
        gallery_feature = extract_feature(model,model_2,dataloaders[gallery_name], which_gallery)
    print(query_feature.shape, gallery_feature.shape)


    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature.numpy(),'query_label':query_label, 'query_path':query_path}
    scipy.io.savemat('pytorch_result.mat', result)

    result = 'result.txt'
    os.system('python evaluate_gpu.py | tee -a %s'%result)