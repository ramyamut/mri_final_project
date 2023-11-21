import torch
import imageio
import glob
import os

class ReconDataset(torch.utils.data.Dataset):
    """takes in a directory for raw kspace data and corresponding reconstructions and creates a dataset for training a reconstruction model
    """

    def __init__(self,
                 kspace_real_dir,
                 kspace_imag_dir,
                 recon_real_dir,
                 recon_imag_dir,
                 eval_mode=False):

        # input data
        self.kspace_real_paths = sorted(glob.glob(f"{kspace_real_dir}/*.png"))
        self.kspace_imag_paths = [os.path.join(kspace_imag_dir, os.path.basename(p)) for p in self.kspace_real_paths]
        self.recon_real_paths = [os.path.join(recon_real_dir, os.path.basename(p)) for p in self.kspace_real_paths]
        self.recon_imag_paths = [os.path.join(recon_imag_dir, os.path.basename(p)) for p in self.kspace_real_paths]

        self.eval_mode = eval_mode

    def __len__(self):
        return len(self.kspace_real_paths)

    def __getitem__(self, idx):

        kspace_real = imageio.imread(self.kspace_real_paths[idx]) / 255
        kspace_imag = imageio.imread(self.kspace_imag_paths[idx]) / 255
        recon_real = imageio.imread(self.recon_real_paths[idx]) / 255
        recon_imag = imageio.imread(self.recon_imag_paths[idx]) / 255
        
        kspace_real_t = torch.tensor(kspace_real).float().unsqueeze(0)
        kspace_imag_t = torch.tensor(kspace_imag).float().unsqueeze(0)
        kspace_t = torch.stack([kspace_real_t, kspace_imag_t], dim=0) * 2 - 1
        recon_real_t = torch.tensor(recon_real).float().unsqueeze(0)
        recon_imag_t = torch.tensor(recon_imag).float().unsqueeze(0)
        recon_t = torch.stack([recon_real_t, recon_imag_t], dim=0) * 2 - 1

        return {
            "kspace": kspace_t,
            "recon": recon_t
        }
