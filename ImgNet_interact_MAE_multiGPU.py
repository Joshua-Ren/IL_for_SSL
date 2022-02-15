# -*- coding: utf-8 -*-
"""
Interaction phase (MAE)
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
import torchvision.transforms as T
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
from data_loader_lmdb import ImageFolderLMDB
import torch.distributed as dist

def parse():
    parser = argparse.ArgumentParser(description='ImageNet-MAE')
    parser.add_argument('--lr', default=1.5e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--batch_size',default=256, type=int)
    parser.add_argument('--seed',default=1086,type=int)
    parser.add_argument('--proj_path',default='Interact_MAE', type=str)
    parser.add_argument('--epochs',default=400, type=int)
    parser.add_argument('--accfreq',default=10, type=int, help='every xx iteration, update acc')
    parser.add_argument('--mask_ratio',default=0.75,type=float)
    parser.add_argument('--run_name',default=None,type=str)
    parser.add_argument('--enable_amp',action='store_true')
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--workers',default=8, type=int)
    parser.add_argument('--record_gap',default=50, type=int)
    parser.add_argument('--dataset',type=str,default='imagenet',help='can be imagenet, tiny')
    parser.add_argument('--modelsize',type=str,default='tiny',help='ViT model size, must be tiny, small or base')
    args = parser.parse_args()
    
    if args.modelsize.lower()=='tiny':
        enc_params = [192, 12, 3, 512]           # dim, depth, heads, mlp_dim
        dec_params = [512, 6]                    # dec_dim, dec_depth
    elif args.modelsize.lower()=='small':
        enc_params = [384, 12, 6, 1024]          # dim, depth, heads, mlp_dim
        dec_params = [512, 6] #[1024, 2]                   # dec_dim, dec_depth
    elif args.modelsize.lower()=='base':
        enc_params = [768, 12, 12, 2048]         # dim, depth, heads, mlp_dim
        dec_params = [512, 6] #[2048, 4]                   # dec_dim, dec_depth
    else:
        print('ViT model size must be tiny, small, or base')
    [args.enc_dim, args.enc_depth, args.enc_heads, args.enc_mlp] = enc_params
    [args.dec_dim, args.dec_depth] = dec_params

    if args.dataset.lower()=='imagenet':
        tmp_kfp=[1000, 256, 256, 16, 1] # k_clas, fill_size, fig_size, patch_size, ds_ratio
    elif args.dataset.lower()=='tiny':
        tmp_kfp=[200, 64, 64, 8, 1]
    else:
        print('dataset must be imagenet or tiny')
    [args.k_clas, args.fill_size, args.fig_size, args.patch_size, args.ds_ratio] = tmp_kfp
    args.patch_num=int(args.fill_size/args.patch_size)
    return args

# =================== Some utils functions ==========================
def _recon_validate(TRACK_TVX, mae,table_key='initial'):
    '''
        For image reconstruction, feed TRACK_TVX to the mae model
        then show the reconstruction and original figure on W&B
    '''
    loss, recon_img_patches = mae(TRACK_TVX)
    recon_imgs = rearrange(recon_img_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                               h=args.patch_size,w=args.patch_size,c=3, p1=args.patch_num,p2=args.patch_num)
    origi_imgs = TRACK_TVX
    wandb_show16imgs(recon_imgs, origi_imgs, table_key=table_key, ds_ratio=args.ds_ratio)

def adjust_learning_rate(args, optimizer, epoch):
    """
        warm up (linearly to lr) 5-10 epoch, then cosine decay to lr_min
    """
    warmup_ep = 10
    lr_min = 1e-7
    lr_start = args.lr
    if epoch<warmup_ep:
        lr_current = lr_min+(lr_start-lr_min)*(epoch)/warmup_ep
    else:
        degree = (epoch-warmup_ep)/(args.epochs-warmup_ep)*np.pi
        lr_current = lr_min+0.5*(lr_start-lr_min)*(1+np.cos(degree))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_current
    return lr_current

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
    encoder = ViT(image_size=args.fig_size, patch_size=args.patch_size, num_classes=args.k_clas,
                  dim=args.enc_dim, depth=args.enc_depth, heads=args.enc_heads, mlp_dim=args.enc_mlp)
    mae = my_MAE(encoder=encoder, masking_ratio=args.mask_ratio, decoder_dim=args.dec_dim, decoder_depth=args.dec_depth)
    if args.sync_bn:
        print("using apex synced BN")
        mae = parallel.convert_syncbn_model(mae)      
    mae.cuda()

    # Scale learning rate based on global batch size
    #args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    optimizer = optim.AdamW(mae.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    if args.enable_amp:
        mae, optimizer = amp.initialize(mae, optimizer, opt_level="O1")
    if args.distributed:
        mae = DDP(mae, delay_allreduce=True)

    # ================== Prepare for the dataloader ===============
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    traindir = os.path.join('/home/sg955/rds/rds-nlp-cdt-VR7brx3H4V8/datasets/ImageNet/', 'train.lmdb')
    train_set = ImageFolderLMDB(
        traindir, T.Compose([T.RandomResizedCrop(args.fig_size), T.RandomHorizontalFlip(),
            T.ToTensor(), normalize, ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    # =================== Initialize wandb ========================
    if args.local_rank==0:
        run_name = wandb_init(proj_name=args.proj_path, run_name=args.run_name, config_args=args)
        #run_name = 'add'
        base_folder = '/home/sg955/GitWS/IL_for_SSL/'
        save_path = base_folder+'results/'+args.proj_path+'/'+args.modelsize+'/'+run_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        TRACK_TVX = wandb_gen_track_x(train_loader)
        TRACK_TVX = TRACK_TVX.cuda()  
        
    # ================= Train the model ===========================
    for g in range(args.epochs):
        if args.local_rank==0:
            _recon_validate(TRACK_TVX, mae,table_key='latest')
        # ----- Do validation only on rank0
        if g%args.record_gap == 0:
            if args.local_rank==0:
                CK_PATH = checkpoint_save_interact(mae, g, save_path)
                _recon_validate(TRACK_TVX, mae,table_key='ep'+str(g))
            if False:#args.distributed:
                dist.barrier()
                # configure map_location properly
                map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
                mae.load_state_dict(
                    torch.load(CK_PATH, map_location=map_location))
        train(train_loader, mae, optimizer, g)
        #torch.cuda.synchronize()    # If also use val_loader, open this, but in interact, no need
        #train_loader.reset()
        #val_loader.reset() 

def train(train_loader, mae, optimizer, g):
    mae.train()
    
    #for i, data in enumerate(train_loader):
    #    x = data[0]["data"]
    for i, (x, _) in enumerate(train_loader):
        x = x.cuda(non_blocking=True)
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
        if i%args.accfreq == 0:
            torch.cuda.synchronize()
    if args.local_rank==0:
        curr_lr = adjust_learning_rate(args, optimizer, g)
        wandb.log({'learn_rate':curr_lr})
        wandb.log({'epoch':g})
if __name__ == '__main__':
    main()






















