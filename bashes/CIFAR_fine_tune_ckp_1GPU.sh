#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p pascal
#SBATCH --time=36:00:00
#SBATCH --job-name=C-finet
#SBATCH --output=./logs/test.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1

source /home/sg955/egg-env/bin/activate

cd /home/sg955/GitWS/IL_for_SSL/

srun python CIFAR_finetune_singleGPU.py --enable_amp \
--run_name 1GPU --dataset cifar100 --modelsize base \
--loadrun basetry_4GPU_1kbs --loadep ep50