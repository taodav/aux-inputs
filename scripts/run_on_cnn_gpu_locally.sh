#!/usr/bin/env bash

# MAKE SURE array here is num_jobs // RUNS_PER_JOB

# get the correct version of both from CC

cd ../  # Go to main project folder

source venv/bin/activate
#export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
##
#RUNS_PER_JOB=3

# The -u means ungrouped - output is ungrouped and printed.
#parallel -j $RUNS_PER_JOB -u < scripts/runs/runs_uf4_comb.txt

#export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

RUNS_PER_JOB=1

UPPER_IDX=$((80 * RUNS_PER_JOB))
LOWER_IDX=$(($UPPER_IDX - $RUNS_PER_JOB + 1))

#SED_STR=""
echo "${LOWER_IDX},${UPPER_IDX}p"

# First we get the RUNS_PER_JOBS lines that we're going to run
TO_RUN=$(sed -n "${LOWER_IDX},${UPPER_IDX}p" scripts/runs/runs_uf8_cnn_lstm_t1.txt)
echo "$TO_RUN"

# The -u means ungrouped - output is ungrouped and printed.
parallel --joblog "$SCRATCH/log/on_cnn_gpu.log" -u ::: "$TO_RUN"
