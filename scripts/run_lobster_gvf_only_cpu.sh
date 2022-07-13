#!/bin/bash

#SBATCH --account=def-amw8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rtao3@ualberta.ca
#SBATCH --error=/home/taodav/scratch/log/uncertainty/2-gvf-%j-%n-%a.err
#SBATCH --output=/home/taodav/scratch/log/uncertainty/2-gvf-%j-%n-%a.out
#SBATCH --cpus-per-task=7
#SBATCH --mem=10G
#SBATCH --time=0-6:00
#SBATCH --array=1-18

# MAKE SURE array here is num_jobs // RUNS_PER_JOB

cd ../  # Go to main project folder
source venv/bin/activate

RUNS_PER_JOB=5
#RUNS_PER_JOB=1

UPPER_IDX=$((SLURM_ARRAY_TASK_ID * RUNS_PER_JOB))
LOWER_IDX=$(($UPPER_IDX - $RUNS_PER_JOB + 1))

# First we get the RUNS_PER_JOBS lines that we're going to run
TO_RUN=$(sed -n "${LOWER_IDX},${UPPER_IDX}p" scripts/runs/lobster_nn_gvf_fixed_sweep.txt)

echo "$TO_RUN"

# The -u means ungrouped - output is ungrouped and printed.
parallel --joblog "$SCRATCH/log/2_gvf.log" -u ::: "$TO_RUN"
