#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

cd ../
source venv/bin/activate
parallel --joblog log/'lobster_nn_sweep.log' -u < 'scripts/runs/lobster_nn_pf_sweep.txt'

