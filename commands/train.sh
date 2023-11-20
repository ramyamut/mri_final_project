#!/bin/bash

#pip install numpy
export PYTHONPATH=/data/vision/polina/users/ramyamut/projects/mri_final_project/

## EXECUTION OF PYTHON CODE:
python /data/vision/polina/users/ramyamut/projects/mri_final_project/scripts/training.py \
  --train_kspace_real_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_train/preproc/kspace_real \
  --train_kspace_imag_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_train/preproc/kspace_imag \
  --train_recon_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_train/preproc/recon \
  --val_kspace_real_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_val/preproc/kspace_real \
  --val_kspace_imag_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_val/preproc/kspace_imag \
  --val_recon_dir /data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_val/preproc/recon \
  --model_dir /data/vision/polina/users/ramyamut/mri_final_project/models/training_debug \
  --batchsize 2