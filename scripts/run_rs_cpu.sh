#!/bin/sh

#SBATCH --account=def-amw8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rtao3@ualberta.ca
#SBATCH --error=/home/taodav/scratch/log/slurm-%j-%n-%a.err
#SBATCH --output=/home/taodav/scratch/log/slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=3G
#SBATCH --time=0-12:00
#SBATCH --array=1-1080

cd ../  # Go to main project folder
source venv/bin/activate

TO_RUN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/runs/runs_rs.txt)
eval $TO_RUN

# The -u means ungrouped - output is ungrouped and printed.
#parallel --joblog ../log/'runs.log' -u < 'scripts/runs/runs.txt'
