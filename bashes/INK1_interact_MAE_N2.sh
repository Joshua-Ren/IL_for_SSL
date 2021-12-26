#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --time=36:00:00
#SBATCH --job-name=mae
#SBATCH --output=./logs/test.txt
#SBATCH --nodes=2
#SBATCH --gres=gpu:4

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1

source /home/sg955/egg-env/bin/activate

cd /home/sg955/GitWS/IL_for_SSL/

srun python -m torch.distributed.launch --nproc_per_node=8 --master_port 10086 INK1_interact_MAE.py --run_name DALI_AMP_8GPU --enable_amp