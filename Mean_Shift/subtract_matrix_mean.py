# This script visualizes diffusion tractograms from six directions
# 
# Author: Hanliang Xu
# Date: July 3, 2023

import os
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

BIOCARD_path = Path('/nfs2/harmonization/raw/BIOCARD_ConnectomeSpecial')
VMAP_path = Path('/nfs2/harmonization/raw/VMAP_ConnectomeSpecial')
BIOCARD_output_path = Path('/nfs2/xuh11/Mean_Shift/Mean_Shift_BIOCARD')
VMAP_output_path = Path('/nfs2/xuh11/Mean_Shift/Mean_Shift_VMAP')

# Load the CSV file
BIOCARD_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalBIOCARD.csv")
VMAP_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalVMAP.csv")

def readCSVfile(filepath):
    with open(filepath, "r") as file:
        return float(file.read())

def applyConnectomeTransformation(inputfile, outputfile, filter):
    array = np.loadtxt(inputfile, delimiter=',')
    filter = np.load(filter)
    array += filter
    array[array < 0] = 0
    np.savetxt(outputfile, array, delimiter=',')

def target_cohorts(cohorts_csv, input_folder, output_folder, file_to_operate_on, filter):
    # Iterate over all rows in finalBIOCARD.csv
    for _, row in cohorts_csv.iterrows():
        sub = row["sub"]
        ses = row["ses"]

        folderpath = input_folder / sub / ses
        print(folderpath)
        for run in folderpath.iterdir():
            if (run.name.startswith('run')):
                if ((sub == "sub-JHU308071" and ses == "ses-150811") or (sub == "sub-JHU666199" and ses == "ses-160503")):
                    input_file_2 = folderpath / "run-2" / file_to_operate_on
                    output_path_2 = output_folder / sub / ses / "run-2"
                    subprocess.run(["mkdir", "-p", output_path_2], check=True)
                    output_file_2 = output_path_2 / file_to_operate_on
                    applyConnectomeTransformation(input_file_2, output_file_2, filter)
                else:
                    input_file_1 = folderpath / "run-1" / file_to_operate_on
                    output_path_1 = output_folder / sub / ses / "run-1"
                    subprocess.run(["mkdir", "-p", output_path_1], check=True)
                    output_file_1 = output_path_1 / file_to_operate_on
                    applyConnectomeTransformation(input_file_1, output_file_1, filter)
                    input_file_2 = folderpath / "run-2" / file_to_operate_on
                    output_path_2 = output_folder / sub / ses / "run-2"
                    subprocess.run(["mkdir", "-p", output_path_2], check=True)
                    output_file_2 = output_path_2 / file_to_operate_on
                    applyConnectomeTransformation(input_file_2, output_file_2, filter)
            else:
                input_file = folderpath / file_to_operate_on
                output_path = output_folder / sub / ses
                subprocess.run(["mkdir", "-p", output_path], check=True)
                output_file = output_path / file_to_operate_on
                applyConnectomeTransformation(input_file, output_file, filter)
            break
    # Now write the collected numbers to a new CSV file
    """
    with open("/nfs2/xuh11/Connectome/Analysis/BIOCARD_modularity.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for number in BIOCARD_modularity:
            writer.writerow([number])  # write each number on a new line
    """


#target_cohorts(BIOCARD_df, BIOCARD_path, BIOCARD_output_path, "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv", "/nfs2/xuh11/Connectome/Analysis/diff_mean_length.npy")
#target_cohorts(BIOCARD_df, BIOCARD_path, BIOCARD_output_path, "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000.csv", "/nfs2/xuh11/Connectome/Analysis/diff_num_streamlines.npy")
target_cohorts(VMAP_df, VMAP_path, VMAP_output_path, "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv", "/nfs2/xuh11/Connectome/Analysis/diff_mean_length.npy")
target_cohorts(VMAP_df, VMAP_path, VMAP_output_path, "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000.csv", "/nfs2/xuh11/Connectome/Analysis/diff_num_streamlines.npy")