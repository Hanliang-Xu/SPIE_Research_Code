import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

BIOCARD_path = Path('/nfs2/harmonization/raw/BIOCARD_ConnectomeSpecial')
VMAP_path = Path('/nfs2/harmonization/raw/VMAP_ConnectomeSpecial')
BIOCARD_CycleGAN = Path('/nfs2/xuh11/Connectome/CycleGan-results/BIOCARD')
VMAP_CycleGAN = Path('/nfs2/xuh11/Connectome/CycleGan-results/VMAP')

# Load the CSV file
BIOCARD_df = pd.read_csv("/nfs2/xuh11/Connectome/CycleGAN-ipynb/pytorch-CycleGAN-and-pix2pix/datasets/test_set_A.csv")
VMAP_df = pd.read_csv("/nfs2/xuh11/Connectome/CycleGAN-ipynb/pytorch-CycleGAN-and-pix2pix/datasets/test_set_B.csv")

def readCSVfile(filepath):
    with open(filepath, "r") as file:
        return float(file.read())

def graph_measure_pipe(path, starts_with_run, graph_measure) :
    file_name = "GraphMeasure_" + graph_measure + ".csv"
    if os.path.isfile(path / "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv") or os.path.isfile(path / "run-1" / "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv"):
        if (starts_with_run):
            return ((readCSVfile(path / "run-1" / file_name) + readCSVfile(path / "run-2" / file_name)) / 2)
        else:
            return (readCSVfile(path / file_name))
    return 99999999999

# Iterate over all rows in finalBIOCARD.csv
def collect_graph_measure(demographics, input_folder, graph_measure):
    result = []
    for _, row in demographics.iterrows():
        sub = row["sub"]
        ses = row["ses"]
        # Construct the path to the number.csv file
        folderpath = input_folder / sub / ses
        for run in folderpath.iterdir():
            if (sub == "sub-JHU666199" and ses == "ses-160503" and input_folder != BIOCARD_CycleGAN):
                result.append(graph_measure_pipe(folderpath / "run-2", False, graph_measure))
            elif (sub == "sub-JHU989472" and ses == "ses-150427" and input_folder != BIOCARD_CycleGAN):
                result.append(graph_measure_pipe(folderpath / "run-1", False, graph_measure))
            elif(run.name.startswith('run')):
                result.append(graph_measure_pipe(folderpath, True, graph_measure))
            else:
                result.append(graph_measure_pipe(folderpath, False, graph_measure))
            break
    return result


BIOCARD_orig_assortativity = collect_graph_measure(BIOCARD_df, BIOCARD_path, "assortativity")
BIOCARD_orig_averagebetweennesscentrality = collect_graph_measure(BIOCARD_df, BIOCARD_path, "averagebetweennesscentrality")
BIOCARD_orig_characteristicpathlength = collect_graph_measure(BIOCARD_df, BIOCARD_path, "characteristicpathlength")
BIOCARD_orig_globalefficiency = collect_graph_measure(BIOCARD_df, BIOCARD_path, "globalefficiency")
BIOCARD_orig_modularity = collect_graph_measure(BIOCARD_df, BIOCARD_path, "modularity")


BIOCARD_mean_std_assortativity = collect_graph_measure(BIOCARD_df, BIOCARD_CycleGAN, "assortativity")
BIOCARD_mean_std_averagebetweennesscentrality = collect_graph_measure(BIOCARD_df, BIOCARD_CycleGAN, "averagebetweennesscentrality")
BIOCARD_mean_std_characteristicpathlength = collect_graph_measure(BIOCARD_df, BIOCARD_CycleGAN, "characteristicpathlength")
BIOCARD_mean_std_globalefficiency = collect_graph_measure(BIOCARD_df, BIOCARD_CycleGAN, "globalefficiency")
BIOCARD_mean_std_modularity = collect_graph_measure(BIOCARD_df, BIOCARD_CycleGAN, "modularity")

VMAP_orig_assortativity = collect_graph_measure(VMAP_df, VMAP_path, "assortativity")
VMAP_orig_averagebetweennesscentrality = collect_graph_measure(VMAP_df, VMAP_path, "averagebetweennesscentrality")
VMAP_orig_characteristicpathlength = collect_graph_measure(VMAP_df, VMAP_path, "characteristicpathlength")
VMAP_orig_globalefficiency = collect_graph_measure(VMAP_df, VMAP_path, "globalefficiency")
VMAP_orig_modularity = collect_graph_measure(VMAP_df, VMAP_path, "modularity")

VMAP_CycleGan_assortativity = collect_graph_measure(VMAP_df, VMAP_CycleGAN, "assortativity")
VMAP_CycleGan_averagebetweennesscentrality = collect_graph_measure(VMAP_df, VMAP_CycleGAN, "averagebetweennesscentrality")
VMAP_CycleGan_characteristicpathlength = collect_graph_measure(VMAP_df, VMAP_CycleGAN, "characteristicpathlength")
VMAP_CycleGan_globalefficiency = collect_graph_measure(VMAP_df, VMAP_CycleGAN, "globalefficiency")
VMAP_CycleGan_modularity = collect_graph_measure(VMAP_df, VMAP_CycleGAN, "modularity")


def generate_plots(orig1, cycleGAN1, orig2, cycleGAN2):
    # Create figure and axis
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(18, 8), sharey=True)

    sns.violinplot(ax=axs[0], y=orig1, color='darkred')
    axs[0].set_title('Original BIOCARD')

    sns.violinplot(ax=axs[1], y=cycleGAN1, color='red')
    axs[1].set_title('CycleGAN BIOCARD')

    #sns.violinplot(ax=axs[2], y=orig2, color='salmon')
    #axs[2].set_title('Combatted BIOCARD')

    sns.violinplot(ax=axs[2], y=orig2, color='darkgreen')
    axs[2].set_title('Original VMAP')

    sns.violinplot(ax=axs[3], y=cycleGAN2, color='mediumseagreen')
    axs[3].set_title('CycleGAN VMAP')

    #sns.violinplot(ax=axs[5], y=mean_sub2, color='lightgreen')
    #axs[5].set_title('Combatted VMAP')

    plt.tight_layout()
    plt.show()

generate_plots(BIOCARD_orig_assortativity, BIOCARD_mean_std_assortativity, VMAP_orig_assortativity, VMAP_CycleGan_assortativity)
generate_plots(BIOCARD_orig_averagebetweennesscentrality, BIOCARD_mean_std_averagebetweennesscentrality, VMAP_orig_averagebetweennesscentrality, VMAP_CycleGan_averagebetweennesscentrality)
generate_plots(BIOCARD_orig_characteristicpathlength, BIOCARD_mean_std_characteristicpathlength, VMAP_orig_characteristicpathlength, VMAP_CycleGan_characteristicpathlength)
generate_plots(BIOCARD_orig_globalefficiency, BIOCARD_mean_std_globalefficiency, VMAP_orig_globalefficiency, VMAP_CycleGan_globalefficiency)
generate_plots(BIOCARD_orig_modularity, BIOCARD_mean_std_modularity, VMAP_orig_modularity, VMAP_CycleGan_modularity)