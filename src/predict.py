# python imports
import os
import numpy as np
import imageio
import glob
import torch

# project imports
from src import networks, utils


def predict(k_space_real_dir,
            k_space_imag_dir,
            recon_real_dir,
            recon_imag_dir,
            output_dir,
            path_model,
            hidden_channels=64,
            n_layers=5,
            layer_type='interleaved'):
    
    # evaluate
    eval_folder = os.path.join(output_dir, "results")
    os.makedirs(eval_folder, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # create network and lightning module
    channels = [hidden_channels] * (n_layers - 1)
    net = networks.Net(
        channels=channels,
        layer_type=layer_type
    ).to(device)
    state_dict = torch.load(path_model, map_location=device)["state_dict"]
    keys = list(state_dict.keys())
    new_state_dict = {}
    for k in keys:
        new_state_dict[k.replace('model.', '')] = state_dict[k]
    net.load_state_dict(new_state_dict, strict=True)
    net.eval()

    path_kspace_real = sorted(glob.glob(f"{k_space_real_dir}/*png*"))
    path_kspace_imag = [os.path.join(k_space_imag_dir, os.path.basename(p)) for p in path_kspace_real]
    assert all([os.path.exists(p) for p in path_kspace_imag])

    path_recon_real = [os.path.join(recon_real_dir, os.path.basename(p)) for p in path_kspace_real]
    path_recon_imag = [os.path.join(recon_imag_dir, os.path.basename(p)) for p in path_kspace_real]
    assert all([os.path.exists(p) for p in path_recon_real])
    assert all([os.path.exists(p) for p in path_recon_imag])
    
    for i in range(len(path_kspace_real)):
        print(f'processing {i+1} of {len(path_kspace_real)}')
        
        kspace_real = imageio.imread(path_kspace_real[i]) / 255
        kspace_imag = imageio.imread(path_kspace_imag[i]) / 255
        recon_real = imageio.imread(path_recon_real[i]) / 255
        recon_imag = imageio.imread(path_recon_imag[i]) / 255
        
        kspace_real_t = torch.tensor(kspace_real).float().unsqueeze(0) * 2 - 1
        kspace_imag_t = torch.tensor(kspace_imag).float().unsqueeze(0) * 2 - 1
        kspace_t = torch.stack([kspace_real_t, kspace_imag_t], dim=0).to(device) # [2, 1, H, W]
        recon_real_t = torch.tensor(recon_real).float().unsqueeze(0) * 2 - 1
        recon_imag_t = torch.tensor(recon_imag).float().unsqueeze(0) * 2 - 1
        recon_t = torch.stack([recon_real_t, recon_imag_t], dim=0).to(device) # [2, 1, H, W]

        # prediction
        with torch.no_grad():
            pred = net(kspace_t.unsqueeze(0), recon_t.unsqueeze(0)).squeeze().detach().cpu() #[2, H, W]
            pred_proc = (pred + 1)/2
            pred_proc = abs(torch.complex(pred_proc[0], pred_proc[1])).numpy()
        
            # save
            output_path = os.path.join(eval_folder, os.path.basename(path_kspace_real[i]))
            final_img = (pred_proc * 255).astype(np.uint8)
            imageio.imwrite(output_path, final_img)