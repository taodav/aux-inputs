#!/bin/bash

#SBATCH --account=rrg-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rtao3@ualberta.ca
#SBATCH --error=/home/taodav/scratch/log/uncertainty/on-cnn-%j-%n-%a.err
#SBATCH --output=/home/taodav/scratch/log/uncertainty/on-cnn-%j-%n-%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=0-12:00
#SBATCH --array=3-34

# MAKE SURE array here is num_jobs // RUNS_PER_JOB

# get the correct version of both from CC
module load gcc python/3.8 cuda/11.4 cudnn/8.2

cd ../  # Go to main project folder
source venv/bin/activate

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

RUNS_PER_JOB=3

UPPER_IDX=$((SLURM_ARRAY_TASK_ID * RUNS_PER_JOB))
LOWER_IDX=$(($UPPER_IDX - $RUNS_PER_JOB + 1))

# First we get the RUNS_PER_JOBS lines that we're going to run
TO_RUN=$(sed -n "${LOWER_IDX},${UPPER_IDX}p" scripts/runs/runs_uf8_cnn.txt)

echo "$TO_RUN"

# The -u means ungrouped - output is ungrouped and printed.
parallel --joblog "$SCRATCH/log/on_cnn_gpu.log" -u ::: "$TO_RUN"
