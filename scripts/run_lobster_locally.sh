#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

cd ../
source venv/bin/activate
parallel -u < 'scripts/runs/lobster_ppo_best.txt'

