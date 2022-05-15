#!/bin/sh

#SBATCH --account=rrg-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rtao3@ualberta.ca
#SBATCH --error=/home/taodav/scratch/log/uncertainty/rs-%j-%n-%a.err
#SBATCH --output=/home/taodav/scratch/log/uncertainty/rs-%j-%n-%a.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=4G
#SBATCH --time=0-12:00
#SBATCH --array=1-150

cd ../  # Go to main project folder
source venv/bin/activate

TO_RUN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/runs/runs_rs_best.txt)
eval $TO_RUN

# The -u means ungrouped - output is ungrouped and printed.
#parallel --joblog ../log/'runs.log' -u < 'scripts/runs/runs.txt'
