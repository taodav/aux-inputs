#!/usr/bin/env bash

cd ../
source venv/bin/activate
parallel --joblog log/'lobster_nn_sweep.log' -u < 'scripts/runs/lobster_nn_pf_sweep.txt'

