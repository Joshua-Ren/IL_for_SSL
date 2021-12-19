#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --time=36:00:00
#SBATCH --job-name=mae
#SBATCH --output=./logs/test.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1

source /home/sg955/egg-env/bin/activate

cd /home/sg955/GitWS/better_supervision/

srun python main_50_to_18_difftemp_table1.py 