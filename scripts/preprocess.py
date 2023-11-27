import sys
from argparse import ArgumentParser
from src.preprocess import preprocess

parser = ArgumentParser()

# ------------------------------------------------- General parameters -------------------------------------------------

# PATHS
parser.add_argument("--raw_data_dir", type=str, dest="raw_data_dir", default=None)
parser.add_argument("--preproc_dir", type=str, dest="preproc_dir", default=None)

# PREPROCESSING PARAMETERS
parser.add_argument("--final_size", type=int, dest="final_size", default=256)

# OTHER

# PRINT ALL ARGUMENTS
print('\nScript name:',  sys.argv[0])
print('\nScript arguments:')
args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))
print('')
preprocess(**vars(args))
