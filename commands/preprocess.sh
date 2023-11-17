#!/bin/bash

# activate virtual environment
#source /data/vision/polina/users/bbillot/miniconda3/bin/activate tf36
export PYTHONPATH=/data/vision/polina/users/ramyamut/projects/mri_final_project

## EXECUTION OF PYTHON CODE:
python /data/vision/polina/users/ramyamut/projects/mri_final_project/scripts/preprocess.py \
  --raw_data_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_test \
  --preproc_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_test/preproc