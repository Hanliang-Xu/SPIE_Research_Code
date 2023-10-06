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
path_quality_check = Path('/nfs2/xuh11/Connectome/QA_VMAP')
output_folder = Path('/nfs2/xuh11/Connectome/QA_VMAP/imagesInOneFolder')
#rater_offsets = [-25, 0, 25]  # how much deviation from the center slice

# for i,offset in enumerate(rater_offsets):
    # print("Start preparing samples for rater{}...".format(str(i+1)))

list_subs = [sub.name for sub in path_quality_check.iterdir() if sub.name.startswith('sub')]
for sub in tqdm(list_subs, total=len(list_subs)):
    sub_folder = path_quality_check / sub
    for session in sub_folder.iterdir():
        for con in session.iterdir():
            if con.name.startswith("CON"):
                output_file_name = sub_folder.name + "_" + session.name + "_" + con.name
                subprocess.run(['cp',con, output_folder])
                subprocess.run(['mv',output_folder/con.name, output_folder/output_file_name])