# This script converts all the connectomes in the VMAP folder to .npy files
# 
# Author: Hanliang Xu
# Date: Jun 7, 2023

import subprocess
import nibabel as nib
#import matplotlib.pyplot as plt
import numpy as np
# from tqdm import tqdm
from pathlib import Path, PurePath
import csv
import pandas as pd

# path of input and output
path_datasets = Path('/nfs2/harmonization/raw/VMAP_ConnectomeSpecial')
path_output = Path('/nfs2/xuh11/npy-VMAP/data')

print("Start converting .csv file to .npy file")
for dataset in path_datasets.iterdir():
    make_folder_depth_one = PurePath.joinpath(path_output, dataset.name)
    subprocess.run(['mkdir', make_folder_depth_one])
    orig_folder_depth_one = PurePath.joinpath(path_datasets, dataset.name)

    for sub in orig_folder_depth_one.iterdir():
        make_folder_depth_two = PurePath.joinpath(make_folder_depth_one, sub.name)
        subprocess.run(['mkdir', make_folder_depth_two])
        orig_folder_depth_two = PurePath.joinpath(orig_folder_depth_one, sub.name)
        for name in orig_folder_depth_two.iterdir():
            if (name.name.startswith("CONNECTOME")) :
                name_without_csv = name.name[:-4]
                make_folder_depth_three = PurePath.joinpath(make_folder_depth_two, name_without_csv)
                arr = np.genfromtxt(str(name), dtype="float64", delimiter=",")
                np.save(make_folder_depth_three, arr)