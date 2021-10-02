#!/usr/bin/env bash

TO_RUN=$(sed -n "1p" runs/runs_rs_prefilled.txt)
cd ../
eval $TO_RUN