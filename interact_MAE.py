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

K_CLAS = 100

parser = argparse.ArgumentParser(description='ImageNet1K-MAE')
parser.add_argument('--lr', default=1.5e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size',default=1024, type=int)
parser.add_argument('--seed',default=10086,type=int)
parser.add_argument('--proj_path',default='Interact_MAE', type=str)
parser.add_argument('--epochs',default=1000, type=int)
parser.add_argument('--mask_ratio',default=0.5,type=float)

args = parser.parse_args()
rnd_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======== Set results saving things ========================
run_name = wandb_init(proj_name=args.proj_path, run_name=None, config_args=args)
#run_name = 'add'
save_path = './results/MAE/run_'+run_name
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
# ======== Get Dataloader and tracking images ===================
train_T=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
val_T =transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=train_T),batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=val_T),batch_size=10000, shuffle=False, drop_last=True)

TRACK_TVX = wandb_gen_track_x(train_loader,val_loader)
TRACK_TVX = TRACK_TVX.to(device)

# ====================== Interaction phase: MAE ===============================
# ---------- Prepare the model, optimizer, scheduler
encoder = ViT(image_size = 32, patch_size = 4, num_classes = K_CLAS,
              dim = 256, depth = 3, heads = 4, mlp_dim = 512)
mae = my_MAE(encoder=encoder, masking_ratio = args.mask_ratio, decoder_dim = 256, decoder_depth=1).to(device)
optimizer = optim.AdamW(mae.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)

# ---------- Record experimental parameters
def _recon_validate(mae,table_key='initial'):
    '''
        For image reconstruction, feed TRACK_TVX to the mae model
        then show the reconstruction and original figure on W&B
    '''
    loss, recon_img_patches = mae(TRACK_TVX)
    recon_imgs = rearrange(recon_img_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                               h=8,w=8,c=3, p1=4,p2=4)
    origi_imgs = TRACK_TVX
    wandb_show16imgs(recon_imgs, origi_imgs, table_key=table_key, ds_ratio=1)


# ---------- Train the model
for g in range(args.epochs):
    for i, (x, _) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to(device)
        loss,_ = mae(x)
        loss.backward()
        optimizer.step() 
        wandb.log({'loss':loss.item()})
    _recon_validate(mae,table_key='latest')
    if g%50 == 0:
        _recon_validate(mae,table_key='epoch_'+str(g))
        checkpoint_save_interact(mae, g, save_path)
    



















