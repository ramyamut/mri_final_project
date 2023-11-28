# python imports
import os
import numpy as np
import imageio
import glob
import torch
from scipy import fft
from skimage.metrics import structural_similarity as ssim

# project imports
from src import networks, utils, subsample

def predict(k_space_dir,
            output_dir,
            path_model,
            subsample_method=None,
            subsample_factor=1,
            hidden_channels=64,
            n_layers=5,
            layer_type='interleaved'):
    
    # evaluate
    eval_folder = os.path.join(output_dir, "results")
    kspace_folder = os.path.join(eval_folder, "kspace")
    img_folder = os.path.join(eval_folder, "img")
    recon_folder = os.path.join(eval_folder, "recon")
    pred_folder = os.path.join(eval_folder, "pred")
    
    for folder in [eval_folder, kspace_folder, img_folder, recon_folder, pred_folder]:
        os.makedirs(folder, exist_ok=True)
        
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
    
    try:
        subsample_method = subsample.__getattribute__(subsample_method)
        subsample_fn = lambda x: subsample_method(x, subsample_factor)
    except:
        subsample_fn = lambda x: x

    path_kspace = sorted(glob.glob(f"{k_space_dir}/*npy*"))
    
    ssim_scores = []
    
    for i in range(len(path_kspace)):
        print(f'processing {i+1} of {len(path_kspace)}')
        
        kspace = np.load(path_kspace[i])
        kspace_real = kspace[0]
        kspace_imag = kspace[1]
        kspace_complex = kspace_real + 1j*kspace_imag
        recon_complex = fft.ifftshift(fft.ifft2(fft.fftshift(kspace_complex)))
        recon = np.stack([recon_complex.real, recon_complex.imag], axis=0)
        kspace_subsampled = subsample_fn(kspace_complex)
        img_complex = fft.ifftshift(fft.ifft2(fft.fftshift(kspace_subsampled)))
        kspace = np.stack([kspace_subsampled.real, kspace_subsampled.imag], axis=0)
        img = np.stack([img_complex.real, img_complex.imag], axis=0)
        
        kspace_t = torch.tensor(utils.normalize_01(kspace), device=device).unsqueeze(1) * 2 - 1
        recon_t = torch.tensor(utils.normalize_01(recon), device=device).unsqueeze(1) * 2 - 1
        img_t = torch.tensor(utils.normalize_01(img), device=device).unsqueeze(1) * 2 - 1
        
        output_fname = os.path.basename(path_kspace[i]).replace(".npy", ".png")

        # prediction
        with torch.no_grad():
            pred = net(kspace_t.unsqueeze(0), img_t.unsqueeze(0)).squeeze().detach().cpu() #[2, H, W]
            pred_proc = (pred + 1)/2
            pred_proc = abs(torch.complex(pred_proc[0], pred_proc[1])).numpy()
        
            # save prediction
            output_path = os.path.join(pred_folder, output_fname)
            final_img = (pred_proc * 255).astype(np.uint8)
            imageio.imwrite(output_path, final_img)
        
        # save img
        output_path = os.path.join(img_folder, output_fname)
        img_proc = img_t.squeeze().detach().cpu()
        img_proc = (img_proc + 1)/2
        img_proc = abs(torch.complex(img_proc[0], img_proc[1])).numpy()
        final_img = (img_proc * 255).astype(np.uint8)
        imageio.imwrite(output_path, final_img)
        
        # save ground truth
        output_path = os.path.join(recon_folder, output_fname)
        recon_proc = recon_t.squeeze().detach().cpu()
        recon_proc = (recon_proc + 1)/2
        recon_proc = abs(torch.complex(recon_proc[0], recon_proc[1])).numpy()
        final_img = (recon_proc * 255).astype(np.uint8)
        imageio.imwrite(output_path, final_img)
        
        # calculate ssim score btw pred & ground truth
        ssim_score = ssim(recon_proc, pred_proc, data_range=1.)
        ssim_scores.append(ssim_score)
        
        # save undersampled kspace
        output_path = os.path.join(kspace_folder, output_fname)
        kspace_proc = kspace_t.squeeze().detach().cpu()
        kspace_proc = (kspace_proc + 1)/2
        kspace_proc = abs(torch.complex(kspace_proc[0], kspace_proc[1])).numpy()
        final_img = (kspace_proc * 255).astype(np.uint8)
        imageio.imwrite(output_path, final_img)

    np.save(os.path.join(eval_folder, "ssim.npy"), ssim_scores)
