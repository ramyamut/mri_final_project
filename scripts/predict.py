import numpy as np
from src.predict import predict
import glob

# inputs
kspace_dir = '/data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_val/preproc/kspace'
path_model = glob.glob(f'/data/vision/polina/users/ramyamut/projects/mri_final_project/models/alternating_equi_4/model*.ckpt')[0] #model or last

# outputs
output_dir = '/data/vision/polina/users/ramyamut/projects/mri_final_project/models/alternating_equi_4/'

# architecture
hidden_channels = 32
n_layers = 5
layer_type = 'alternating'

predict(kspace_dir,
        output_dir,
        path_model,
        subsample_method='subsample_rows_equi',
        subsample_factor=4,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        layer_type=layer_type)
