from torchvision import transforms
from datasets.Dataloader_University import Sampler_University,Dataloader_University,train_collate_fn
from datasets.random_erasing import RandomErasing
from datasets.autoaugment import ImageNetPolicy, CIFAR10Policy
import torch
import argparse


def make_dataset(opt):          #dataloaders为双层list，[0][0],[1][0]为图片，[0][1],[1][1]为所属标签，一一对应关系
    ######################################################################
    # Load Data
    # ---------
    #

    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_satellite_list = [
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(opt.pad, padding_mode='edge'),
        transforms.RandomAffine(90),
        transforms.RandomCrop((opt.h, opt.w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if opt.erasing_p > 0:
        transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

    if opt.color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                       hue=0)] + transform_train_list
        transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                           hue=0)] + transform_satellite_list

    if opt.DA:
        transform_train_list = [ImageNetPolicy()] + transform_train_list

    print(transform_train_list)
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
        'satellite': transforms.Compose(transform_satellite_list)}


    # image_datasets = {}
    # image_datasets['satellite'] = datasets.ImageFolder(os.path.join(data_dir, 'satellite'),
    #                                                    data_transforms['satellite'])
    # image_datasets['street'] = datasets.ImageFolder(os.path.join(data_dir, 'street'),
    #                                                 data_transforms['train'])
    # image_datasets['drone'] = datasets.ImageFolder(os.path.join(data_dir, 'drone'),
    #                                                data_transforms['train'])
    # image_datasets['google'] = datasets.ImageFolder(os.path.join(data_dir, 'google'),
    #                                                 data_transforms['train'])
    #
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
    #                                               shuffle=True, num_workers=opt.num_worker, pin_memory=True)
    #                # 8 workers may work faster
    #                for x in ['satellite', 'street', 'drone', 'google']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'drone']}
    # class_names = image_datasets['street'].classes
    # print(dataset_sizes)
    # return dataloaders,class_names,dataset_sizes

    # custom Dataset

    image_datasets = Dataloader_University(opt.data_dir,transforms=data_transforms)
    samper = Sampler_University(image_datasets,batchsize=opt.batchsize,sample_num=opt.sample_num)
    dataloaders =torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,sampler=samper,num_workers=opt.num_worker, pin_memory=True,collate_fn=train_collate_fn)
    # print(len(dataloaders))
    dataset_sizes = {x: len(image_datasets)*opt.sample_num for x in ['satellite', 'drone']}
    class_names = image_datasets.cls_names
    return dataloaders,class_names,dataset_sizes


def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name',default='FSRA', type=str, help='output model name')
    parser.add_argument('--data_dir',default='E:\\University-Release\\train',type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true', help='use all training data' )
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
    parser.add_argument('--num_worker', default=4,type=int, help='' )
    parser.add_argument('--batchsize', default=2, type=int, help='batchsize')
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--h', default=256, type=int, help='height')
    parser.add_argument('--w', default=256, type=int, help='width')
    parser.add_argument('--views', default=2, type=int, help='the number of views')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
    parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation' )
    parser.add_argument('--share', action='store_true',default=True, help='share weight between different view' )
    parser.add_argument('--fp16', action='store_true',default=False, help='use float16 instead of float32, which will save about 50% memory' )
    parser.add_argument('--autocast', action='store_true',default=True, help='use mix precision' )
    parser.add_argument('--block', default=2, type=int, help='')
    parser.add_argument('--kl_loss', action='store_true',default=False, help='kl_loss' )
    parser.add_argument('--triplet_loss', default=0.3, type=float, help='')
    parser.add_argument('--sample_num', default=2, type=float, help='num of repeat sampling' )
    parser.add_argument('--num_epochs', default=120, type=int, help='' )
    parser.add_argument('--steps', default=[70,110], type=int, help='' )
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 256), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]


    transform_train_list ={"satellite": transforms.Compose(transform_train_list),
                            "drone":transforms.Compose(transform_train_list)}
    opt = get_parse()
    dataloaders, class_names, dataset_sizes = make_dataset(opt)
    for data_s, data_d in dataloaders:
        print(data_s[0].size(),data_s[1],data_d[0].size(),data_d[1])






