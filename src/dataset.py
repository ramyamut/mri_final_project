import torch
import imageio
import glob
import os
import numpy as np
import scipy.fft as fft

from src import utils, subsample

class ReconDataset(torch.utils.data.Dataset):
    """takes in a directory for raw kspace data and corresponding reconstructions and creates a dataset for training a reconstruction model
    """

    def __init__(self,
                 kspace_dir,
                 subsample_method=None,
                 subsample_factor=1,
                 eval_mode=False):

        # input data
        self.kspace_paths = sorted(glob.glob(f"{kspace_dir}/*.npy"))
        try:
            subsample_method = subsample.__getattribute__(subsample_method)
            self.subsample_fn = lambda x: subsample_method(x, subsample_factor)
        except:
            self.subsample_fn = lambda x: x

        self.eval_mode = eval_mode

    def __len__(self):
        return len(self.kspace_paths)
    
    def __getitem__(self, idx):

        kspace = np.load(self.kspace_paths[idx])
        kspace_real = kspace[0]
        kspace_imag = kspace[1]
        kspace_complex = kspace_real + 1j*kspace_imag
        recon_complex = fft.ifftshift(fft.ifft2(fft.fftshift(kspace_complex)))
        recon = np.stack([recon_complex.real, recon_complex.imag], axis=0)
        kspace_subsampled = self.subsample_fn(kspace_complex)
        img_complex = fft.ifftshift(fft.ifft2(fft.fftshift(kspace_subsampled)))
        kspace = np.stack([kspace_subsampled.real, kspace_subsampled.imag], axis=0)
        img = np.stack([img_complex.real, img_complex.imag], axis=0)
        
        kspace_t = torch.tensor(utils.normalize_01(kspace)).unsqueeze(1) * 2 - 1
        recon_t = torch.tensor(utils.normalize_01(recon)).unsqueeze(1) * 2 - 1
        img_t = torch.tensor(utils.normalize_01(img)).unsqueeze(1) * 2 - 1

        return {
            "kspace": kspace_t,
            "img": img_t,
            "recon": recon_t
        }
        