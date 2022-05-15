#!/bin/sh

#SBATCH --account=def-amw8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rtao3@ualberta.ca
#SBATCH --error=/home/taodav/scratch/log/uncertainty/cw-%j-%n-%a.err
#SBATCH --output=/home/taodav/scratch/log/uncertainty/cw-%j-%n-%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=3G
#SBATCH --time=0-24:00
#SBATCH --array=2,3,5,6,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30

cd ../  # Go to main project folder
source venv/bin/activate

TO_RUN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/runs/runs_compass_lstm_best.txt)
eval $TO_RUN

# The -u means ungrouped - output is ungrouped and printed.
#parallel --joblog ../log/'runs.log' -u < 'scripts/runs/runs.txt'
