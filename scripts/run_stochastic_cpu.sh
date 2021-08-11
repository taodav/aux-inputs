#!/bin/sh

#SBATCH --account=def-amw8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rtao3@ualberta.ca
#SBATCH --error=/home/taodav/scratch/log/slurm-%j-%n-%a.err
#SBATCH --output=/home/taodav/scratch/log/slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --time=2-0:00

cd ../  # Go to main project folder
source venv/bin/activate

# The -u means ungrouped - output is ungrouped and printed.
parallel --joblog ../log/'runs.log' -u < 'scripts/runs/runs_stochastic.txt'
