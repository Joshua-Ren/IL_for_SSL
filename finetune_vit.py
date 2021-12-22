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

K_CLAS = 100

parser = argparse.ArgumentParser(description='ImageNet1K-Finetune')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size',default=128, type=int)
parser.add_argument('--seed',default=1,type=int)
parser.add_argument('--proj_path',default='Finetune_MAE', type=str)
parser.add_argument('--epochs',default=1, type=int)
parser.add_argument('--checkpoint_path',default='MAE/run_ethereal-moon-13/checkpoint',type=str)

args = parser.parse_args()
rnd_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======== Set results saving things ========================
run_name = wandb_init(proj_name=args.proj_path, run_name=None, config_args=args)
#run_name = 'add'
save_path = './results/FineTune/run_'+run_name
if not os.path.exists(save_path):
    os.makedirs(save_path)

# ======== Get Dataloader and tracking images ===================
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

# ====================== Fine-tune phase: 1k classification ===================
# ---------- Prepare (load) the model, optimizer, scheduler
encoder = ViT(image_size = 32, patch_size = 8, num_classes = K_CLAS,
              dim = 1024, depth = 6, heads = 8, mlp_dim = 2048)
encoder.to(device)
chkp_name = 'encoder_ep0.pt'
chkp_path = os.path.join('./results/', args.checkpoint_path,chkp_name)
encoder.load_state_dict(torch.load(chkp_path))

optimizer = optim.AdamW(encoder.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)


def get_validation(model, data_loader):
    # Here only calculate validation acc and loss
    model.eval()
    loss_all = 0
    corr_cnt, sample_cnt = 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(train_loader):
            print(i,end='-')
            x, y = x.to(device), y.to(device)
            hid = model(x)
            pred = hid.argmax(1)
            loss_all += nn.CrossEntropyLoss()(hid, y)
            sample_cnt += x.shape[0]
            corr_cnt += pred.eq(y.view_as(pred)).sum()
    loss = loss_all/(i+1)
    acc = corr_cnt/sample_cnt
    model.train()
    return acc.cpu(), loss.cpu()

# ---------- Train the model
results = {'tacc':[],'tloss':[],'vacc':[],'vloss':[],'bestg_ac':[],'bestg_lo':[]}
loss_all = 0
corr_cnt, sample_cnt = 0, 0
vacc_max, vloss_min = 0, 10
bestg_ac, bestg_lo = 0, 0

for g in range(args.epochs):
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        hid = encoder(x)
        loss = nn.CrossEntropyLoss()(hid, y)
        loss.backward()
        optimizer.step() 
        wandb.log({'loss':loss.item()})
        # ----- Calculate train accuracy acc and loss
        loss_all += loss.item()
        pred = hid.argmax(1)
        sample_cnt += x.shape[0]
        corr_cnt += pred.eq(y.view_as(pred)).sum()
    # -------- At the end of each epoch
    scheduler.step()
    vacc, vloss = get_validation(encoder, val_loader)
    if vacc >= vacc_max:
        bestg_ac = g
    if vloss <= vloss_min:
        bestg_lo = g
    results['bestg_ac'].append(bestg_ac)
    results['bestg_lo'].append(bestg_lo)
    results['tacc'].append(corr_cnt/sample_cnt)
    results['tloss'].append(loss_all/(i+1))
    results['vacc'].append(vacc)
    results['vloss'].append(vloss)
    wandb_record_results(results, g)
    if g%50 == 0:
        checkpoint_save_pretrain(encoder, g, save_path)
        