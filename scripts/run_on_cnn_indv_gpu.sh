#!/bin/bash

#SBATCH --account=def-amw8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rtao3@ualberta.ca
#SBATCH --error=/home/taodav/scratch/log/uncertainty/on-cnn-%j-%n-%a.err
#SBATCH --output=/home/taodav/scratch/log/uncertainty/on-cnn-%j-%n-%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=9G
#SBATCH --time=0-12:00
#SBATCH --array=82,85,88,95,97,98,99

# get the correct version of both from CC
module load gcc python/3.8 cuda/11.4 cudnn/8.2

cd ../  # Go to main project folder
source venv/bin/activate

TO_RUN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/runs/runs_uf8_cnn.txt)
eval $TO_RUN
