#!/usr/bin/env bash

# MAKE SURE array here is num_jobs // RUNS_PER_JOB

# get the correct version of both from CC

cd ../  # Go to main project folder
source venv/bin/activate

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

RUNS_PER_JOB=3

# The -u means ungrouped - output is ungrouped and printed.
parallel -j $RUNS_PER_JOB -u < scripts/runs/runs_uf2_cnn_lstm.txt
