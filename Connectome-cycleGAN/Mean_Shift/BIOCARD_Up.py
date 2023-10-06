# Shift matrices of BIOCARD by adding (VMAP_average - BIOCARD_average)
# 
# Author: Hanliang Xu
# Date: July 17, 2023

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

BIOCARD_path = Path('/nfs2/harmonization/raw/BIOCARD_ConnectomeSpecial')
BIOCARD_output_path = Path('/nfs2/xuh11/Mean_Shift_BIOCARD_Up')
BIOCARD_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalBIOCARD.csv")

def readCSVfile(filepath):
    with open(filepath, "r") as file:
        return float(file.read())

def applyConnectomeTransformation(inputfile, outputfile, mean1, mean2):
    array = np.loadtxt(inputfile, delimiter=',')
    mean1 = np.load(mean1)
    mean2 = np.load(mean2)

    array -= mean1
    array += mean2

    array[array < 0] = 0
    np.savetxt(outputfile, array, delimiter=',')

def target_cohorts(cohorts_csv, input_folder, output_folder, file_to_operate_on, mean1, mean2):
    # Iterate over all rows in finalBIOCARD.csv
    for _, row in cohorts_csv.iterrows():
        sub = row["sub"]
        ses = row["ses"]

        folderpath = input_folder / sub / ses
        print(folderpath)
        for run in folderpath.iterdir():
            if (run.name.startswith('run')):
                if (sub == "sub-JHU666199" and ses == "ses-160503"):
                    input_file_2 = folderpath / "run-2" / file_to_operate_on
                    output_path_2 = output_folder / sub / ses / "run-2"
                    subprocess.run(["mkdir", "-p", output_path_2], check=True)
                    output_file_2 = output_path_2 / file_to_operate_on
                    applyConnectomeTransformation(input_file_2, output_file_2, mean1, mean2)
                elif (sub == "sub-JHU989472" and ses == "ses-150427"):
                    input_file_1 = folderpath / "run-1" / file_to_operate_on
                    output_path_1 = output_folder / sub / ses / "run-1"
                    subprocess.run(["mkdir", "-p", output_path_1], check=True)
                    output_file_1 = output_path_1 / file_to_operate_on
                    applyConnectomeTransformation(input_file_1, output_file_1, mean1, mean2)
                else:
                    input_file_1 = folderpath / "run-1" / file_to_operate_on
                    output_path_1 = output_folder / sub / ses / "run-1"
                    subprocess.run(["mkdir", "-p", output_path_1], check=True)
                    output_file_1 = output_path_1 / file_to_operate_on
                    applyConnectomeTransformation(input_file_1, output_file_1, mean1, mean2)
                    input_file_2 = folderpath / "run-2" / file_to_operate_on
                    output_path_2 = output_folder / sub / ses / "run-2"
                    subprocess.run(["mkdir", "-p", output_path_2], check=True)
                    output_file_2 = output_path_2 / file_to_operate_on
                    applyConnectomeTransformation(input_file_2, output_file_2, mean1, mean2)
            else:
                input_file = folderpath / file_to_operate_on
                output_path = output_folder / sub / ses
                subprocess.run(["mkdir", "-p", output_path], check=True)
                output_file = output_path / file_to_operate_on
                applyConnectomeTransformation(input_file, output_file, mean1, mean2)
            break

target_cohorts(BIOCARD_df, BIOCARD_path, BIOCARD_output_path, "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv", "/nfs2/xuh11/Connectome/Analysis/BIOCARD_mean_length_mean.npy", "/nfs2/xuh11/Connectome/Analysis/VMAP_mean_length_mean.npy")
target_cohorts(BIOCARD_df, BIOCARD_path, BIOCARD_output_path, "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000.csv", "/nfs2/xuh11/Connectome/Analysis/BIOCARD_num_streamlines_mean.npy", "/nfs2/xuh11/Connectome/Analysis/VMAP_num_streamlines_mean.npy")