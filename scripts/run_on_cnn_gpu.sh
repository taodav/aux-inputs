#!/bin/bash

#SBATCH --account=def-amw8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rtao3@ualberta.ca
#SBATCH --error=/home/taodav/scratch/log/uncertainty/on-cnn-%j-%n-%a.err
#SBATCH --output=/home/taodav/scratch/log/uncertainty/on-cnn-%j-%n-%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=9G
#SBATCH --time=0-12:00
#SBATCH --array=1-40

# MAKE SURE array here is num_jobs // RUNS_PER_JOB

# get the correct version of both from CC
module load cuda/11.4 cudnn

cd ../  # Go to main project folder
source venv/bin/activate

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

RUNS_PER_JOB=3

UPPER_IDX=$((SLURM_ARRAY_TASK_ID * RUNS_PER_JOB))
LOWER_IDX=$(($UPPER_IDX - $RUNS_PER_JOB + 1))

SED_STR=""

# First we get the RUNS_PER_JOBS lines that we're going to run
TO_RUN=$(sed -n "${LOWER_IDX},${UPPER_IDX}p" scripts/runs/runs_uf2m_cnn.txt)
echo "$TO_RUN"

# The -u means ungrouped - output is ungrouped and printed.
parallel --joblog "$SCRATCH/log/on_cnn_gpu.log" -u ::: "$TO_RUN"
