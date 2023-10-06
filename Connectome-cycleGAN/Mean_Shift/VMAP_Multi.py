# Shift matrices of BIOCARD by adding (VMAP_average - BIOCARD_average)
# 
# Author: Hanliang Xu
# Date: July 17, 2023

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

VMAP_path = Path('/nfs2/harmonization/raw/VMAP_ConnectomeSpecial')
VMAP_output_path = Path('/nfs2/xuh11/Mean_Shift_VMAP_Multi')
VMAP_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalVMAP.csv")

BIOCARD_Mean_Length_Mean = "/nfs2/xuh11/Connectome/Analysis/BIOCARD_mean_length_mean.npy"
VMAP_Mean_Length_Mean = "/nfs2/xuh11/Connectome/Analysis/VMAP_mean_length_mean.npy"
BIOCARD_Num_Streamlines_Mean = "/nfs2/xuh11/Connectome/Analysis/BIOCARD_num_streamlines_mean.npy"
VMAP_Num_Streamlines_Mean = "/nfs2/xuh11/Connectome/Analysis/VMAP_num_streamlines_mean.npy"

def readCSVfile(filepath):
    with open(filepath, "r") as file:
        return float(file.read())

def applyConnectomeTransformation(inputfile, outputfile, mean1, mean2, multi_factor):
    array = np.loadtxt(inputfile, delimiter=',')

    mean1 = np.load(mean1)
    mean2 = np.load(mean2)

    array -= mean1
    array += mean2
    array *= multi_factor

    array[array < 0] = 0
    np.savetxt(outputfile, array, delimiter=',')

def target_cohorts(cohorts_csv, input_folder, output_folder, file_to_operate_on, mean1, mean2, multi_factor):
    # Iterate over all rows in finalBIOCARD.csv
    for _, row in cohorts_csv.iterrows():
        sub = row["sub"]
        ses = row["ses"]
        folderpath = input_folder / sub / ses
        print(folderpath)
        input_file = folderpath / file_to_operate_on
        output_path = output_folder / sub / ses
        subprocess.run(["mkdir", "-p", output_path], check=True)
        output_file = output_path / file_to_operate_on
        applyConnectomeTransformation(input_file, output_file, mean1, mean2, multi_factor)

target_cohorts(VMAP_df, VMAP_path, VMAP_output_path, "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv", VMAP_Mean_Length_Mean, BIOCARD_Mean_Length_Mean, 0.5)
target_cohorts(VMAP_df, VMAP_path, VMAP_output_path, "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000.csv", VMAP_Num_Streamlines_Mean, BIOCARD_Num_Streamlines_Mean, 0.9)