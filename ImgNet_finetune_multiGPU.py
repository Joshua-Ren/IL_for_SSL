# -*- coding: utf-8 -*-
"""
Usually, do not save checkpoints in fine-tune, we only need the accuracy.
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
from data_loader_lmdb import ImageFolderLMDB
import os
import argparse
import random
from utils import *
import copy
from vit_pytorch import ViT
from my_MAE import my_MAE
from einops import rearrange, repeat
import torch.distributed as dist

def parse():
    parser = argparse.ArgumentParser(description='ImageNet-Finetune')
    parser.add_argument('--scratch',action='store_true',help='train from scratch')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--batch_size',default=256, type=int)
    parser.add_argument('--seed',default=10086,type=int)
    parser.add_argument('--proj_path',default='Finetune_ImgNet', type=str)
    parser.add_argument('--epochs',default=100, type=int)
    parser.add_argument('--accfreq',default=10, type=int, help='every xx iteration, update acc')
    parser.add_argument('--run_name',default=None,type=str)
    parser.add_argument('--enable_amp',action='store_true')
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--workers',default=4, type=int)
    parser.add_argument('--dataset',type=str,default='tiny',help='can be imagenet, tiny')
    parser.add_argument('--modelsize',type=str,default='tiny',help='ViT model size, must be tiny, small or base')
    parser.add_argument('--loadrun',type=str,default='basetry_4GPU_1kbs')
    parser.add_argument('--loadep',type=str,default='ep0')
    args = parser.parse_args()
    #/home/sg955/GitWS/IL_for_SSL/results/Interact_MAE/base/basetry_4GPU_1kbs/checkpoint
    # For example ../Interact_MAE/tiny/tinytry_4GPU/checkpoint/encoder_ep0.pt
    base_folder = '/home/sg955/GitWS/IL_for_SSL/'
    base_path = base_folder + 'results/Interact_MAE/'
    base_file = 'encoder_'+args.loadep+'.pt'
    args.load_ckpt_path = os.path.join(base_path, args.modelsize.lower(),
                           args.loadrun,'checkpoint', base_file)
    args.run_name =  args.dataset+'_'+args.modelsize+'_'+ args.loadep+'__'+args.run_name                
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
        tmp_kfp=[1000, 256, 256, 16, 4] # k_clas, fill_size, fig_size, patch_size, ds_ratio
    elif args.dataset.lower()=='tiny':
        tmp_kfp=[200, 64, 64, 8, 1]
    else:
        print('dataset must be imagenet or tiny')
    [args.k_clas, args.fill_size, args.fig_size, args.patch_size, args.ds_ratio] = tmp_kfp
    args.patch_num=int(args.fig_size/args.patch_size)
    return args
    
# =================== Some utils functions ==========================
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt
    
def adjust_learning_rate(args, optimizer, epoch):
    """
    warm up (linearly to lr) 5-10 epoch, then cosine decay to lr_min
    """
    warmup_ep = 5
    lr_min = 1e-6
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
    # Here we create mae and only use encoder to make finetune (as checkpoint is saved as mae)
    encoder = ViT(image_size=args.fig_size, patch_size=args.patch_size, num_classes=args.k_clas,
                  dim=args.enc_dim, depth=args.enc_depth, heads=args.enc_heads, mlp_dim=args.enc_mlp)
    mae = my_MAE(encoder=encoder, masking_ratio=0.75, decoder_dim=args.dec_dim, decoder_depth=args.dec_depth)
    if args.sync_bn:
        print("using apex synced BN")
        mae = parallel.convert_syncbn_model(mae)      
    
    if not args.scratch:
        ckp = ckp_converter(torch.load(args.load_ckpt_path))
        mae.load_state_dict(ckp)

    # Scale learning rate based on global batch size
    encoder.cuda()
    args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    optimizer = optim.AdamW(encoder.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    if args.enable_amp:
        encoder, optimizer = amp.initialize(encoder, optimizer, opt_level="O1")
    if args.distributed:
        encoder = DDP(encoder, delay_allreduce=True)
 
    # ================== Prepare for the dataloader ===============
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    traindir = os.path.join('/home/sg955/rds/rds-nlp-cdt-VR7brx3H4V8/datasets/ImageNet/', 'train.lmdb')
    valdir = os.path.join('/home/sg955/rds/rds-nlp-cdt-VR7brx3H4V8/datasets/ImageNet/', 'val.lmdb')
    train_set = ImageFolderLMDB(
        traindir, T.Compose([T.RandomResizedCrop(args.fig_size), T.RandomHorizontalFlip(),
            T.ToTensor(), normalize, ]))
    val_set = ImageFolderLMDB(
        valdir, T.Compose([ T.Resize(args.fill_size), T.CenterCrop(args.fig_size),
            T.ToTensor(),normalize, ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)    
    # =================== Initialize wandb ========================
    if args.local_rank==0:
        run_name = wandb_init(proj_name=args.proj_path, run_name=args.run_name, config_args=args)
        #run_name = 'add'
        save_path = base_folder+'results/'+args.proj_path+'/'+run_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # ================= Train the model ===========================
    for g in range(args.epochs):
        train(train_loader, encoder, optimizer, g)
        _accuracy_validate(val_loader, encoder)
        torch.cuda.synchronize()    # If also use val_loader, open this, but in interact, no need


def train(train_loader, encoder, optimizer, g):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    encoder.train()

    for i, (x, y) in enumerate(train_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        # compute output, for encoder, we need cls token to get hid
        hid = encoder(x)
        loss = nn.CrossEntropyLoss()(hid, y)
        optimizer.zero_grad()
        if args.enable_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if i%args.accfreq == 0:
            prec1, prec5 = accuracy(hid.data, y, topk=(1, 5))
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data
            losses.update(reduced_loss.item(), x.size(0))
            top1.update(prec1.item(), x.size(0))
            top5.update(prec5.item(), x.size(0))   
            torch.cuda.synchronize()
            if args.local_rank==0:
                wandb.log({'loss':loss.item()})
    if args.local_rank==0:
        curr_lr = adjust_learning_rate(args, optimizer, g)
        wandb.log({'epoch':g})
        wandb.log({'train_loss':losses.avg})
        wandb.log({'train_top1':top1.avg})
        wandb.log({'train_top5':top5.avg})
        wandb.log({'learn_rate':curr_lr})
    
def _accuracy_validate(val_loader, encoder):
    '''
        Calculate validation accuracy, support multi-GPU
    '''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    encoder.eval()

    for i, (x, y) in enumerate(val_loader):
        x = x.cuda(args.gpu, non_blocking=True)
        y = y.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.no_grad():
            hid = encoder(x)
            loss = nn.CrossEntropyLoss()(hid, y)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(hid.data, y, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(reduced_loss.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))
        top5.update(prec5.item(), x.size(0))
    if args.local_rank==0:
        wandb.log({'valid_loss':losses.avg})
        wandb.log({'valid_top1':top1.avg})
        wandb.log({'valid_top5':top5.avg})
if __name__ == '__main__':
    main()        