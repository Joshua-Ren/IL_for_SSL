#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --time=36:00:00
#SBATCH --job-name=mae
#SBATCH --output=./logs/test.txt
#SBATCH --nodes=2
#SBATCH --gres=gpu:4

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1

source /home/sg955/egg-env/bin/activate

cd /home/sg955/GitWS/IL_for_SSL/

srun python INK1_interact_MAE.py --run_name DALI_AMP_N2 --enable_amp