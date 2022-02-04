import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import warnings
warnings.filterwarnings('ignore')

#LOCAL_PATH = 'E:\DATASET\tiny-imagenet-200'

@pipeline_def
def create_dali_pipeline(dataset, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):

    if dataset.lower()=='imagenet':
        DATA_PATH = '/home/sg955/rds/rds-nlp-cdt-VR7brx3H4V8/datasets/ImageNet/'
        if is_training:
            data_dir = os.path.join(DATA_PATH, 'train')
        else:
            data_dir = os.path.join(DATA_PATH, 'val')
    elif dataset.lower()=='tiny':
        DATA_PATH = '/home/sg955/rds/hpc-work/tiny-imagenet-200/'
        if is_training:
            data_dir = os.path.join(DATA_PATH, 'train')
        else:
            data_dir = os.path.join(DATA_PATH, 'val')
    
    images, labels = fn.readers.file(file_root=data_dir, shard_id=shard_id, num_shards=num_shards,
                                    random_shuffle=is_training, pad_last_batch=True, name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    if is_training:
        images = fn.decoders.image_random_crop(images, device=decoder_device, output_type=types.RGB, random_aspect_ratio=[0.8, 1.25], 
                                                random_area=[0.1, 1.0], num_attempts=100)
        images = fn.resize(images, device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images, device=decoder_device, output_type=types.RGB)
        images = fn.resize(images, device=dali_device, resize_x=crop, resize_y=crop, mode="not_smaller", interp_type=types.INTERP_TRIANGULAR)
        #images = fn.resize(images, device=dali_device, size=size, mode="not_smaller", interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(), dtype=types.FLOAT, output_layout="CHW",
                                    crop=(crop, crop),mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                    std=[0.229 * 255,0.224 * 255,0.225 * 255], mirror=mirror)
    labels = labels.gpu()
    return images, labels




if __name__ == '__main__':

    # iteration of PyTorch dataloader
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dst = datasets.ImageFolder(IMG_DIR, transform_train)
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=2048, shuffle=True, pin_memory=True, num_workers=8)
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dst = datasets.ImageFolder(IMG_DIR, transform_val)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=2000, shuffle=False, pin_memory=True, num_workers=8)

    print('[PyTorch] start iterate test dataloader')
    start = time.time()
    for i, (x,y) in enumerate(train_loader):
        if i%5==0:
            print(i,end='-')
        images = x.cuda(non_blocking=True)
        labels = y.cuda(non_blocking=True)
    end = time.time()
    test_time = end-start
    print('[PyTorch] end test dataloader iteration')
    # print('[PyTorch] iteration time: %fs [train],  %fs [test]' % (train_time, test_time))
    print('[PyTorch] iteration time: %fs [test]' % (test_time))


    pipe = create_dali_pipeline(batch_size=2048, num_threads=8, device_id=0, seed=12, data_dir=IMG_DIR,
                                crop=224, size=256, dali_cpu=False, shard_id=0, num_shards=1, is_training=True)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    pipe = create_dali_pipeline(batch_size=2000, num_threads=8, device_id=0, seed=12, data_dir=IMG_DIR,
                                crop=256, size=256, dali_cpu=True, shard_id=0, num_shards=1, is_training=False)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    print('[DALI-GPU] start iterate train dataloader')
    start = time.time()
    for i, data in enumerate(train_loader):
        if i%5==0:
            print(i,end='-')
        images = data[0]['data'].cuda()
        labels = data[0]['label'].cuda()
    end = time.time()
    test_time = end-start
    print('[DALI-GPU] iteration time: %fs [test]' % (test_time))


    print('[DALI-cpu] start iterate val dataloader')
    start = time.time()
    for i, data in enumerate(val_loader):
        if i%5==0:
            print(i,end='-')
        images = data[0]['data'].cuda()
        labels = data[0]['label'].cuda()
    end = time.time()
    test_time = end-start
    print('[DALI-cpu] iteration time: %fs [test]' % (test_time))
