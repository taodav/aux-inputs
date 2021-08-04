#!/bin/sh

#SBATCH --account=def-amw8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rtao3@ualberta.ca
#SBATCH --error=/home/taodav/scratch/log/slurm-%j-%n-%a.err
#SBATCH --output=/home/taodav/scratch/log/slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=40
#SBATCH --mem=40G
#SBATCH --time=1-0:00

cd ../  # Go to main project folder
source venv/bin/activate

# The -u means ungrouped - output is ungrouped and printed.
parallel --joblog ../log/'runs.log' -u < 'scripts/runs/runs.txt'
