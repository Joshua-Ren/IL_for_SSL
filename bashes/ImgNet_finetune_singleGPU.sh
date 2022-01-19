#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --time=36:00:00
#SBATCH --job-name=finetune
#SBATCH --output=./logs/test.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:4

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1

source /home/sg955/egg-env/bin/activate

cd /home/sg955/GitWS/IL_for_SSL/

srun python -m torch.distributed.launch --nproc_per_node=4 --master_port 10086 ImgNet_finetune_multiGPU.py --enable_amp \
--run_name finetune_1GPU_ep400 --dataset tiny --modelsize tiny \
--loadrun tinytry_1GPU --loadep ep400