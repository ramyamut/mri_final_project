# python imports
import os
import glob
import numpy as np
import cv2
import h5py
import scipy.fft as fft

from src.utils import crop_center, normalize_01

def process(data_arr, resize, raw_path, slice_idx, output_dir):
    norm = normalize_01(data_arr)
    resized = cv2.resize(norm, dsize=(resize, resize), interpolation=cv2.INTER_LINEAR)
    subj = os.path.basename(raw_path).split(".h5")[0]
    output_path = os.path.join(output_dir, f"{subj}_slice_{slice_idx}.png")
    cv2.imwrite(output_path, resized)

def preprocess(raw_data_dir,
             preproc_dir,
             final_size,
             recon_crop_size,
             skip_recon=False):
    
    raw_data_paths = sorted(glob.glob(f"{raw_data_dir}/*.h5"))
    kspace_dir = os.path.join(preproc_dir, "kspace")
    recon_dir = os.path.join(preproc_dir, "recon")
    
    for dir in [preproc_dir, kspace_dir, recon_dir]:
        os.makedirs(dir, exist_ok=True)
    
    for i, path in enumerate(raw_data_paths):
        print(f"PROCESSING FILE {i+1} of {len(raw_data_paths)}")
        
        # LOAD KSPACE DATA
        data = h5py.File(path, 'r')["kspace"]
        kspace = np.array(data)
        
        # CREATE IMAGES FOR EACH SLICE AND SAVE
        for slice in range(len(kspace)):
            process(kspace[slice], final_size, path, slice, kspace_dir)
            
            if not skip_recon:
                recon = fft.ifftshift(fft.ifft2(fft.fftshift(kspace[slice])))
                recon = crop_center(recon, recon_crop_size, recon_crop_size)
                process(recon, final_size, path, slice, recon_dir)