import sys
from argparse import ArgumentParser

from src.training import training

parser = ArgumentParser()

# ------------------------------------------------- General parameters -------------------------------------------------

# PATHS
parser.add_argument("--train_kspace_real_dir", type=str, dest="train_kspace_real_dir", default=None)
parser.add_argument("--train_kspace_imag_dir", type=str, dest="train_kspace_imag_dir", default=None)
parser.add_argument("--train_recon_dir", type=str, dest="train_recon_dir", default=None)
parser.add_argument("--val_kspace_real_dir", type=str, dest="val_kspace_real_dir", default=None)
parser.add_argument("--val_kspace_imag_dir", type=str, dest="val_kspace_imag_dir", default=None)
parser.add_argument("--val_recon_dir", type=str, dest="val_recon_dir", default=None)
parser.add_argument("--model_dir", type=str, dest="model_dir", default=None)

# ---------------------------------------------- Augmentation parameters -----------------------------------------------

# OUTPUT
parser.add_argument("--batchsize", type=int, dest="batchsize", default=1)

# -------------------------------------------- UNet architecture parameters --------------------------------------------
parser.add_argument("--hidden_channels", type=int, dest="hidden_channels", default=64)
parser.add_argument("--num_layers", type=int, dest="num_layers", default=5)

# ------------------------------------------------- Training parameters ------------------------------------------------

# GENERAL
parser.add_argument("--lr", type=float, dest="lr", default=1e-4)
parser.add_argument("--weight_decay", type=float, dest="weight_decay", default=0.)
parser.add_argument("--n_epochs", type=int, dest="n_epochs", default=1000)
parser.add_argument("--checkpoint", type=str, dest="checkpoint", default=None)

# PRINT ALL ARGUMENTS
print('\nScript name:',  sys.argv[0])
print('\nScript arguments:')
args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))
print('')
training(**vars(args))
