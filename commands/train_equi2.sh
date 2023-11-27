#!/bin/bash

#pip install numpy
export PYTHONPATH=/data/vision/polina/users/ramyamut/projects/mri_final_project/

## EXECUTION OF PYTHON CODE:
python /data/vision/polina/users/ramyamut/projects/mri_final_project/scripts/training.py \
  --train_kspace_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_train/preproc/kspace \
  --val_kspace_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_val/preproc/kspace \
  --subsample_method subsample_rows_equi \
  --subsample_factor 2 \
  --model_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/models/interleaved_equi_2 \
  --batchsize 2