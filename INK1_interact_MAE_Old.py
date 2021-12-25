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
parser.add_argument('--batch_size',default=2048, type=int)
parser.add_argument('--seed',default=10086,type=int)
parser.add_argument('--proj_path',default='INK1_Interact_MAE', type=str)
parser.add_argument('--epochs',default=1000, type=int)
parser.add_argument('--mask_ratio',default=0.5,type=float)
parser.add_argument('--run_name',default=None,type=str)

args = parser.parse_args()
rnd_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======== Set results saving things ========================
run_name = wandb_init(proj_name=args.proj_path, run_name=args.run_name, config_args=args)
#run_name = 'add'
save_path = './results/INK1_MAE/run_'+run_name
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
# ======== Get Dataloader and tracking images ===================
DATA_PATH = '/home/sg955/rds/hpc-work/ImageNet/'
traindir = os.path.join(DATA_PATH, 'val')
valdir = os.path.join(DATA_PATH, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

train_dataset = torchvision.datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

val_dataset = torchvision.datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
TRACK_TVX = wandb_gen_track_x_old(train_loader,val_loader)
TRACK_TVX = TRACK_TVX.to(device)

# ====================== Interaction phase: MAE ===============================
# ---------- Prepare the model, optimizer, scheduler
encoder = ViT(image_size = 224, patch_size = 16, num_classes = 1000,
              dim = 1024, depth = 6, heads = 8, mlp_dim = 2048)
mae = my_MAE(encoder=encoder, masking_ratio = 0.75, decoder_dim = 512, decoder_depth=1).to(device)
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
                               h=14,w=14,c=3, p1=16,p2=16)
    origi_imgs = TRACK_TVX
    wandb_show16imgs(recon_imgs, origi_imgs, table_key=table_key, ds_ratio=4)

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
        #checkpoint_save_interact(mae, g, save_path)