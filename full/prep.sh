#!/bin/bash

# Inputs

in_dir=$1
gpu=$2

# T1

source /home-local/dt1/code/venv/bin/activate
bash prep_T1.sh $in_dir $gpu
deactivate

# Diffusion

source ~/Apps/scilpy/venv/bin/activate
bash prep_dwmri.sh $in_dir
deactivate

# PyTorch

source /home-local/dt1/code/venv/bin/activate
python prep_pt.py $in_dir 1000000 1000
deactivate
