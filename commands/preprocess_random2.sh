#!/bin/bash

export PYTHONPATH=/data/vision/polina/users/ramyamut/projects/mri_final_project

## EXECUTION OF PYTHON CODE:
python /data/vision/polina/users/ramyamut/projects/mri_final_project/scripts/preprocess.py \
  --raw_data_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_train \
  --preproc_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_train/preproc_random_2 \
  --subsample subsample_rows_random \
  --subsample_factor 2

python /data/vision/polina/users/ramyamut/projects/mri_final_project/scripts/preprocess.py \
  --raw_data_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_val \
  --preproc_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_val/preproc_random_2 \
  --subsample subsample_rows_random \
  --subsample_factor 2