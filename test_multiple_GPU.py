import torch
import argparse
import os
from utils import *
from apex import amp
# FOR DISTRIBUTED: (can also use torch.nn.parallel.DistributedDataParallel instead)
from apex.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--lr', default=1.5e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size',default=1024, type=int)
parser.add_argument('--seed',default=10086,type=int)
parser.add_argument('--proj_path',default='INK1_Interact_MAE', type=str)
parser.add_argument('--epochs',default=1000, type=int)
parser.add_argument('--mask_ratio',default=0.5,type=float)
parser.add_argument('--run_name',default=None,type=str)
parser.add_argument('--enable_amp',action='store_true')
args = parser.parse_args()


# FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
# the 'WORLD_SIZE' environment variable will also be set automatically.
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

torch.backends.cudnn.benchmark = True

if args.local_rank==0:
    wandb_init(proj_name='INK1_Interact_MAE', run_name='test_multipleGPU', config_args=args)

N, D_in, D_out = 64, 1024, 16

# Each process receives its own batch of "fake input data" and "fake target data."
# The "training loop" in each process just uses this fake batch over and over.
# https://github.com/NVIDIA/apex/tree/master/examples/imagenet provides a more realistic
# example of distributed data sampling for both training and validation.
x = torch.randn(N, D_in, device='cuda')
y = torch.randn(N, D_out, device='cuda')

model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if args.distributed:
    # FOR DISTRIBUTED:  After amp.initialize, wrap the model with
    # apex.parallel.DistributedDataParallel.
    model = DistributedDataParallel(model)
    # torch.nn.parallel.DistributedDataParallel is also fine, with some added args:
    # model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                   device_ids=[args.local_rank],
    #                                                   output_device=args.local_rank)

loss_fn = torch.nn.MSELoss()

for t in range(500000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
    if args.local_rank==0:
        wandb.log({'loss':loss.item()})
if args.local_rank == 0:
    print("final loss = ", loss)