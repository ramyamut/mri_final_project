import torch
import imageio
import glob
import os

class ReconDataset(torch.utils.data.Dataset):
    """takes in a directory for raw kspace data and corresponding reconstructions and creates a dataset for training a reconstruction model
    """

    def __init__(self,
                 kspace_dir,
                 recon_dir,
                 eval_mode=False):

        # input data
        self.kspace_paths = sorted(glob.glob(f"{kspace_dir}/*.png"))
        self.recon_paths = [os.path.join(recon_dir, os.path.basename(p)) for p in self.kspace_paths]

        self.eval_mode = eval_mode

    def __len__(self):
        return len(self.kspace_paths)

    def __getitem__(self, idx):

        kspace = imageio.imread(self.kspace_paths[idx])
        recon = imageio.imread(self.recon_paths[idx])
        
        kspace_t = torch.tensor(kspace).float() * 2 - 1
        recon_t = torch.tensor(recon).float() * 2 - 1

        return {
            "kspace": kspace_t,
            "recon": recon_t
        }