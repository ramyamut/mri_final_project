import numpy as np
from src.predict import predict
import glob

# inputs
kspace_real_dir = '/data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_val/preproc/kspace_real'
kspace_imag_dir = '/data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_val/preproc/kspace_imag'
recon_real_dir = '/data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_val/preproc/recon_real'
recon_imag_dir = '/data/vision/polina/users/ramyamut/projects/mri_final_project/data/singlecoil_val/preproc/recon_imag'
path_model = glob.glob(f'/data/vision/polina/users/ramyamut/projects/mri_final_project/models/training_debug_alt/model*.ckpt')[0] #model or last

# outputs
output_dir = '/data/vision/polina/users/ramyamut/projects/mri_final_project/models/training_debug_alt'

# architecture
hidden_channels = 32
n_layers = 5
layer_type = 'alternating'

predict(kspace_real_dir,
        kspace_imag_dir,
        recon_real_dir,
        recon_imag_dir,
        output_dir,
        path_model,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        layer_type=layer_type)