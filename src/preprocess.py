# python imports
import os
import glob
import numpy as np
import imageio
import h5py
import scipy.fft as fft

from src.utils import crop_center

def process(data_arr, raw_path, slice_idx, output_dir):
    data_arr = np.stack([data_arr.real, data_arr.imag], axis=0)
    subj = os.path.basename(raw_path).split(".h5")[0]
    output_path = os.path.join(output_dir, f"{subj}_slice_{slice_idx}.npy")
    np.save(output_path, data_arr)

def preprocess(raw_data_dir,
             preproc_dir,
             final_size):
    
    raw_data_paths = sorted(glob.glob(f"{raw_data_dir}/*.h5"))
    kspace_dir = os.path.join(preproc_dir, "kspace")
    
    for dir in [preproc_dir, kspace_dir]:
        os.makedirs(dir, exist_ok=True)
    
    for i, path in enumerate(raw_data_paths):
        path = raw_data_paths[i]
        print(f"PROCESSING FILE {i+1} of {len(raw_data_paths)}")
        
        # LOAD KSPACE DATA
        try:
            data = h5py.File(path, 'r')["kspace"]
            kspace = np.array(data)
        except:
            continue
        
        # CREATE IMAGES FOR EACH SLICE AND SAVE
        for slice in range(len(kspace)):
            kspace_crop = crop_center(kspace[slice], final_size, final_size)
            process(kspace_crop, path, slice, kspace_dir)
