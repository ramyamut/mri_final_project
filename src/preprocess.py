# python imports
import os
import glob
import numpy as np
import imageio
import h5py
import scipy.fft as fft

from src.utils import crop_center, normalize_01

def process(data_arr, raw_path, slice_idx, output_dir):
    norm = normalize_01(data_arr)
    subj = os.path.basename(raw_path).split(".h5")[0]
    output_path = os.path.join(output_dir, f"{subj}_slice_{slice_idx}.png")
    final_img = (norm * 255).astype(np.uint8)
    imageio.imwrite(output_path, final_img)

def preprocess(raw_data_dir,
             preproc_dir,
             final_size,
             skip_recon=False):
    
    raw_data_paths = sorted(glob.glob(f"{raw_data_dir}/*.h5"))
    kspace_real_dir = os.path.join(preproc_dir, "kspace_real")
    kspace_imag_dir = os.path.join(preproc_dir, "kspace_imag")
    recon_dir = os.path.join(preproc_dir, "recon")
    
    for dir in [preproc_dir, kspace_real_dir, kspace_imag_dir, recon_dir]:
        os.makedirs(dir, exist_ok=True)
    
    #for i, path in enumerate(raw_data_paths):
    # skip 768 and 769
    for i in range(770, len(raw_data_paths)):
        path = raw_data_paths[i]
        print(f"PROCESSING FILE {i+1} of {len(raw_data_paths)}")
        
        # LOAD KSPACE DATA
        data = h5py.File(path, 'r')["kspace"]
        kspace = np.array(data)
        
        # CREATE IMAGES FOR EACH SLICE AND SAVE
        for slice in range(len(kspace)):
            kspace_crop = crop_center(kspace[slice], final_size, final_size)
            process(kspace_crop.real, path, slice, kspace_real_dir)
            process(kspace_crop.imag, path, slice, kspace_imag_dir)
            
            if not skip_recon:
                recon = fft.ifftshift(fft.ifft2(fft.fftshift(kspace_crop)))
                recon = abs(recon)
                process(recon, path, slice, recon_dir)
