# This script visualizes diffusion tractograms from six directions
# 
# Author: Hanliang Xu
# Date: July 3, 2023

import subprocess
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path

# path of input and output
path_datasets = Path('/nfs2/harmonization/raw/VMAP_ConnectomeSpecial')
path_quality_check = Path('/nfs2/xuh11/Connectome/QA_VMAP')

#rater_offsets = [-25, 0, 25]  # how much deviation from the center slice

# for i,offset in enumerate(rater_offsets):
    # print("Start preparing samples for rater{}...".format(str(i+1)))

list_subs = [sub.name for sub in path_datasets.iterdir() if sub.name.startswith('sub-2')]
for sub in tqdm(list_subs, total=len(list_subs)):
    sub_folder = path_datasets / sub

    # Screenshot output directory
    path_output_sub = path_quality_check / sub
    subprocess.run(['mkdir','-p', path_output_sub])
    # print("Start working on {}. \nSaving screenshots to {}".format(sub, path_output_folder))
    
    for session in sub_folder.iterdir():
        path_output_ses = path_output_sub / session.name
        subprocess.run(['mkdir','-p',path_output_ses])
        for con in session.iterdir():
            if con.name.startswith("CON"):
                img_array = np.genfromtxt(con, delimiter=',')
                plt.imshow(img_array)
                img_name = con.name[:-4] + '.png'
                plt.savefig(path_output_ses / img_name)