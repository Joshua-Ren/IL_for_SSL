# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 21:44:23 2021
@author: YIREN
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 17:55:08 2021
@author: YIREN
"""
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import argparse
import random
from utils import *
import copy
from vit_pytorch import ViT
from my_MAE import my_MAE
from einops import rearrange, repeat
from data_loader import ZipImageNetFolder
from data_loader_DALI import *


K_CLAS = 1000

def parse():
    parser = argparse.ArgumentParser(description='ImageNet1K-MAE')
    parser.add_argument('--lr', default=1.5e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batch_size',default=1024, type=int)
    parser.add_argument('--seed',default=10086,type=int)
    parser.add_argument('--proj_path',default='INK1_Interact_MAE', type=str)
    parser.add_argument('--epochs',default=1000, type=int)
    parser.add_argument('--mask_ratio',default=0.5,type=float)
    parser.add_argument('--run_name',default=None,type=str)
    parser.add_argument('--enable_amp',action='store_true')
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--workers',default=4, type=int)
    parser.add_argument('--record_gap',default=50, type=int)
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    #parser.add_argument('--enable_distribute',action='store_true')
    args = parser.parse_args()
    return args

# =================== Some utils functions ==========================
def _recon_validate(TRACK_TVX, mae,table_key='initial'):
    '''
        For image reconstruction, feed TRACK_TVX to the mae model
        then show the reconstruction and original figure on W&B
    '''
    loss, recon_img_patches = mae(TRACK_TVX)
    recon_imgs = rearrange(recon_img_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                               h=14,w=14,c=3, p1=16,p2=16)
    origi_imgs = TRACK_TVX
    wandb_show16imgs(recon_imgs, origi_imgs, table_key=table_key, ds_ratio=4)

def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# ======================== Main and Train ==========================

def main():
    global args
    args = parse()
    rnd_seed(args.seed)
    # ================= Prepare for distributed training =====
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1    
    if args.enable_amp or args.distributed or args.sync_bn:
        global DDP, amp, optimizers, parallel
        from apex.parallel import DistributedDataParallel as DDP
        from apex import amp, optimizers, parallel
    cudnn.benchmark = True
    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    args.total_batch_size = args.world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # ================== Create the model: mae ==================
    encoder = ViT(image_size = 224, patch_size = 16, num_classes = 1000,
                  dim = 1024, depth = 6, heads = 8, mlp_dim = 2048)
    mae = my_MAE(encoder=encoder, masking_ratio = 0.75, decoder_dim = 512, decoder_depth=1)
    if args.sync_bn:
        print("using apex synced BN")
        mae = parallel.convert_syncbn_model(mae)      
    mae.cuda()
    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    optimizer = optim.AdamW(mae.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    if args.enable_amp:
        mae, optimizer = amp.initialize(mae, optimizer, opt_level="O1")
    if args.distributed:
        mae = DDP(mae, delay_allreduce=True)
    
    # ================== Prepare for the dataloader ===============
    DATA_PATH = '/home/sg955/rds/hpc-work/ImageNet/'
    traindir = os.path.join(DATA_PATH, 'val')
    valdir = os.path.join(DATA_PATH, 'val')
    pipe = create_dali_pipeline(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank,
                                seed=12+args.local_rank, data_dir=traindir, crop=224, size=256, dali_cpu=False,
                                shard_id=args.local_rank, num_shards=args.world_size, is_training=True)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    
    pipe = create_dali_pipeline(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank,
                                seed=12+args.local_rank, data_dir=valdir, crop=224, size=256, dali_cpu=False,
                                shard_id=args.local_rank, num_shards=args.world_size, is_training=False)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    
    # =================== Initialize wandb ========================
    if args.local_rank==0:
        run_name = wandb_init(proj_name=args.proj_path, run_name=args.run_name, config_args=args)
        #run_name = 'add'
        save_path = './results/INK1_MAE/run_'+run_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        TRACK_TVX = wandb_gen_track_x(train_loader,val_loader)
        TRACK_TVX = TRACK_TVX.cuda()  
        
    # ================= Train the model ===========================
    for g in range(args.epochs):
        train(train_loader, mae, optimizer, g)
        
        # ----- Do validation only on rank0
        if True:#args.local_rank==0:
            _recon_validate(TRACK_TVX, mae,table_key='latest')
            if g%args.record_gap == 0:
                _recon_validate(TRACK_TVX, mae,table_key='epoch_'+str(g))
                #checkpoint_save_interact(mae, g, save_path)
        
        train_loader.reset()
        val_loader.reset() 


def train(train_loader, mae, optimizer, g):
    losses = AverageMeter()
    mae.train()
    
    for i, data in enumerate(train_loader):
        x = data[0]["data"]
        train_loader_len = int(math.ceil(train_loader._size / args.batch_size))
        adjust_learning_rate(optimizer, g, i, train_loader_len)
        # compute output
        loss,_ = mae(x)
        optimizer.zero_grad()
        if args.enable_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if args.local_rank==0:
            wandb.log({'loss':loss.item()})
        if i%args.print_freq == 0:
            torch.cuda.synchronize()

if __name__ == '__main__':
    main()






















