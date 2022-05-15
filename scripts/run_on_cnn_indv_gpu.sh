#!/bin/bash

#SBATCH --account=def-amw8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rtao3@ualberta.ca
#SBATCH --error=/home/taodav/scratch/log/uncertainty/on-cnn-%j-%n-%a.err
#SBATCH --output=/home/taodav/scratch/log/uncertainty/on-cnn-%j-%n-%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=5G
#SBATCH --time=0-6:00
#SBATCH --array=4,5,6,7,8,9,10,11,12,16,17,18,19,20,21,29,30

# get the correct version of both from CC
module load gcc python/3.8 cuda/11.4 cudnn/8.2

cd ../  # Go to main project folder
source venv/bin/activate

TO_RUN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/runs/runs_uf8_cnn.txt)
eval $TO_RUN
