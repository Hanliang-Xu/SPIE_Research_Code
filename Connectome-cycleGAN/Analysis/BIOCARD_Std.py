# Shift matrices of BIOCARD by adding (VMAP_average - BIOCARD_average)
# 
# Author: Hanliang Xu
# Date: July 17, 2023

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

BIOCARD_path = Path('/nfs2/harmonization/raw/BIOCARD_ConnectomeSpecial')
BIOCARD_output_path = Path('/nfs2/xuh11/Mean_Shift_BIOCARD_Std')
BIOCARD_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalBIOCARD.csv")

BIOCARD_Mean_Length_Mean = "/nfs2/xuh11/Connectome/Analysis/BIOCARD_mean_length_mean.npy"
VMAP_Mean_Length_Mean = "/nfs2/xuh11/Connectome/Analysis/VMAP_mean_length_mean.npy"
BIOCARD_Mean_Length_Std = "/nfs2/xuh11/Connectome/Analysis/BIOCARD_mean_length_std.npy"
VMAP_Mean_Length_Std = "/nfs2/xuh11/Connectome/Analysis/VMAP_mean_length_std.npy"
BIOCARD_Num_Streamlines_Mean = "/nfs2/xuh11/Connectome/Analysis/BIOCARD_num_streamlines_mean.npy"
VMAP_Num_Streamlines_Mean = "/nfs2/xuh11/Connectome/Analysis/VMAP_num_streamlines_mean.npy"
BIOCARD_Num_Streamlines_Std = "/nfs2/xuh11/Connectome/Analysis/BIOCARD_num_streamlines_std.npy"
VMAP_Num_Streamlines_Std = "/nfs2/xuh11/Connectome/Analysis/VMAP_num_streamlines_std.npy"

def readCSVfile(filepath):
    with open(filepath, "r") as file:
        return float(file.read())

def applyConnectomeTransformation(inputfile, outputfile, mean1, mean2, std1, std2):
    array = np.loadtxt(inputfile, delimiter=',')

    mean1 = np.load(mean1)
    mean2 = np.load(mean2)
    std1 = np.load(std1)
    std2 = np.load(std2)

    array -= mean1
    np.divide(array, std1, out=array, where=std1!=0)
    array *= std2
    array += mean2

    array[array < 0] = 0
    np.savetxt(outputfile, array, delimiter=',')

def target_cohorts(cohorts_csv, input_folder, output_folder, file_to_operate_on, mean1, mean2, std1, std2):
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
                    applyConnectomeTransformation(input_file_2, output_file_2, mean1, mean2, std1, std2)
                elif (sub == "sub-JHU989472" and ses == "ses-150427"):
                    input_file_1 = folderpath / "run-1" / file_to_operate_on
                    output_path_1 = output_folder / sub / ses / "run-1"
                    subprocess.run(["mkdir", "-p", output_path_1], check=True)
                    output_file_1 = output_path_1 / file_to_operate_on
                    applyConnectomeTransformation(input_file_1, output_file_1, mean1, mean2, std1, std2)
                else:
                    input_file_1 = folderpath / "run-1" / file_to_operate_on
                    output_path_1 = output_folder / sub / ses / "run-1"
                    subprocess.run(["mkdir", "-p", output_path_1], check=True)
                    output_file_1 = output_path_1 / file_to_operate_on
                    if ((sub == "sub-JHU830375" and ses == "ses-171212")):
                        input_file_1 = "/nfs2/xuh11/ConnectomeSpecialOnRAMBAM/temp-sub-JHU830375_ses-171212_run-1/" + file_to_operate_on
                    applyConnectomeTransformation(input_file_1, output_file_1, mean1, mean2, std1, std2)
                    input_file_2 = folderpath / "run-2" / file_to_operate_on
                    output_path_2 = output_folder / sub / ses / "run-2"
                    subprocess.run(["mkdir", "-p", output_path_2], check=True)
                    output_file_2 = output_path_2 / file_to_operate_on
                    if ((sub == "sub-JHU423254" and ses == "ses-190102")):
                        input_file_2 = "/nfs2/xuh11/ConnectomeSpecialOnRAMBAM/temp-sub-JHU423254_ses-190102_run-2/" + file_to_operate_on
                    applyConnectomeTransformation(input_file_2, output_file_2, mean1, mean2, std1, std2)
            else:
                input_file = folderpath / file_to_operate_on
                output_path = output_folder / sub / ses
                subprocess.run(["mkdir", "-p", output_path], check=True)
                output_file = output_path / file_to_operate_on
                applyConnectomeTransformation(input_file, output_file, mean1, mean2, std1, std2)
            break

target_cohorts(BIOCARD_df, BIOCARD_path, BIOCARD_output_path, "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv", BIOCARD_Mean_Length_Mean, VMAP_Mean_Length_Mean, BIOCARD_Mean_Length_Std, VMAP_Mean_Length_Std)
target_cohorts(BIOCARD_df, BIOCARD_path, BIOCARD_output_path, "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000.csv", BIOCARD_Num_Streamlines_Mean, VMAP_Num_Streamlines_Mean, BIOCARD_Num_Streamlines_Std, VMAP_Num_Streamlines_Std)