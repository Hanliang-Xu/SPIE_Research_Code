import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from util import *

BIOCARD_path = Path('/nfs2/harmonization/raw/BIOCARD_ConnectomeSpecial')
VMAP_path = Path('/nfs2/harmonization/raw/VMAP_ConnectomeSpecial')
BIOCARD_MEAN = Path('/nfs2/xuh11/Mean_Shift_BIOCARD_Up')
BIOCARD_MEAN_ComBat = Path('/nfs2/xuh11/ComBat_BIOCARD')
VMAP_MEAN = Path('/nfs2/xuh11/Mean_Shift_VMAP_Down')
VMAP_MEAN_ComBat = Path('/nfs2/xuh11/ComBat_VMAP')
BIOCARD_CycleGAN = Path('/nfs2/xuh11/Connectome/CycleGan-results/BIOCARD')
VMAP_CycleGAN = Path('/nfs2/xuh11/Connectome/CycleGan-results/VMAP')

# Load the CSV file
BIOCARD_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalBIOCARD.csv")
VMAP_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalVMAP.csv")
BIOCARD_df_CycleGAN = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalBIOCARD.csv")
VMAP_df_CycleGAN = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalVMAP.csv")
Combined = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/matchVMAPwithBIOCARD(sameSexCog0_9Age).csv")



BIOCARD_orig_assortativity = collect_graph_measure(BIOCARD_df, BIOCARD_path, "assortativity")
BIOCARD_orig_averagebetweennesscentrality = collect_graph_measure(BIOCARD_df, BIOCARD_path, "averagebetweennesscentrality")
BIOCARD_orig_characteristicpathlength = collect_graph_measure(BIOCARD_df, BIOCARD_path, "characteristicpathlength")
BIOCARD_orig_globalefficiency = collect_graph_measure(BIOCARD_df, BIOCARD_path, "globalefficiency")
BIOCARD_orig_modularity = collect_graph_measure(BIOCARD_df, BIOCARD_path, "modularity")

BIOCARD_mean_assortativity = collect_graph_measure(BIOCARD_df, BIOCARD_MEAN, "assortativity")
BIOCARD_mean_averagebetweennesscentrality = collect_graph_measure(BIOCARD_df, BIOCARD_MEAN, "averagebetweennesscentrality")
BIOCARD_mean_characteristicpathlength = collect_graph_measure(BIOCARD_df, BIOCARD_MEAN, "characteristicpathlength")
BIOCARD_mean_globalefficiency = collect_graph_measure(BIOCARD_df, BIOCARD_MEAN, "globalefficiency")
BIOCARD_mean_modularity = collect_graph_measure(BIOCARD_df, BIOCARD_MEAN, "modularity")

BIOCARD_mean_std_assortativity = collect_graph_measure(BIOCARD_df, BIOCARD_MEAN_ComBat, "assortativity")
BIOCARD_mean_std_averagebetweennesscentrality = collect_graph_measure(BIOCARD_df, BIOCARD_MEAN_ComBat, "averagebetweennesscentrality")
BIOCARD_mean_std_characteristicpathlength = collect_graph_measure(BIOCARD_df, BIOCARD_MEAN_ComBat, "characteristicpathlength")
BIOCARD_mean_std_globalefficiency = collect_graph_measure(BIOCARD_df, BIOCARD_MEAN_ComBat, "globalefficiency")
BIOCARD_mean_std_modularity = collect_graph_measure(BIOCARD_df, BIOCARD_MEAN_ComBat, "modularity")

BIOCARD_CycleGan_assortativity = collect_graph_measure(BIOCARD_df, BIOCARD_CycleGAN, "assortativity")
BIOCARD_CycleGan_averagebetweennesscentrality = collect_graph_measure(BIOCARD_df, BIOCARD_CycleGAN, "averagebetweennesscentrality")
BIOCARD_CycleGan_characteristicpathlength = collect_graph_measure(BIOCARD_df, BIOCARD_CycleGAN, "characteristicpathlength")
BIOCARD_CycleGan_globalefficiency = collect_graph_measure(BIOCARD_df, BIOCARD_CycleGAN, "globalefficiency")
BIOCARD_CycleGan_modularity = collect_graph_measure(BIOCARD_df, BIOCARD_CycleGAN, "modularity")

VMAP_orig_assortativity = collect_graph_measure(VMAP_df, VMAP_path, "assortativity")
VMAP_orig_averagebetweennesscentrality = collect_graph_measure(VMAP_df, VMAP_path, "averagebetweennesscentrality")
VMAP_orig_characteristicpathlength = collect_graph_measure(VMAP_df, VMAP_path, "characteristicpathlength")
VMAP_orig_globalefficiency = collect_graph_measure(VMAP_df, VMAP_path, "globalefficiency")
VMAP_orig_modularity = collect_graph_measure(VMAP_df, VMAP_path, "modularity")

VMAP_mean_assortativity = collect_graph_measure(VMAP_df, VMAP_MEAN, "assortativity")
VMAP_mean_averagebetweennesscentrality = collect_graph_measure(VMAP_df, VMAP_MEAN, "averagebetweennesscentrality")
VMAP_mean_characteristicpathlength = collect_graph_measure(VMAP_df, VMAP_MEAN, "characteristicpathlength")
VMAP_mean_globalefficiency = collect_graph_measure(VMAP_df, VMAP_MEAN, "globalefficiency")
VMAP_mean_modularity = collect_graph_measure(VMAP_df, VMAP_MEAN, "modularity")

VMAP_mean_std_assortativity = collect_graph_measure(VMAP_df, VMAP_MEAN_ComBat, "assortativity")
VMAP_mean_std_averagebetweennesscentrality = collect_graph_measure(VMAP_df, VMAP_MEAN_ComBat, "averagebetweennesscentrality")
VMAP_mean_std_characteristicpathlength = collect_graph_measure(VMAP_df, VMAP_MEAN_ComBat, "characteristicpathlength")
VMAP_mean_std_globalefficiency = collect_graph_measure(VMAP_df, VMAP_MEAN_ComBat, "globalefficiency")
VMAP_mean_std_modularity = collect_graph_measure(VMAP_df, VMAP_MEAN_ComBat, "modularity")

VMAP_CycleGan_assortativity = collect_graph_measure(VMAP_df, VMAP_CycleGAN, "assortativity")
VMAP_CycleGan_averagebetweennesscentrality = collect_graph_measure(VMAP_df, VMAP_CycleGAN, "averagebetweennesscentrality")
VMAP_CycleGan_characteristicpathlength = collect_graph_measure(VMAP_df, VMAP_CycleGAN, "characteristicpathlength")
VMAP_CycleGan_globalefficiency = collect_graph_measure(VMAP_df, VMAP_CycleGAN, "globalefficiency")
VMAP_CycleGan_modularity = collect_graph_measure(VMAP_df, VMAP_CycleGAN, "modularity")
import statistics
print(statistics.mean(BIOCARD_orig_globalefficiency))
print(statistics.mean(BIOCARD_mean_globalefficiency))
print(statistics.mean(BIOCARD_CycleGan_globalefficiency))
print(statistics.mean(BIOCARD_mean_std_globalefficiency))
print(statistics.mean(VMAP_orig_globalefficiency))
print(statistics.mean(VMAP_mean_globalefficiency))
print(statistics.mean(VMAP_CycleGan_globalefficiency))
print(statistics.mean(VMAP_mean_std_globalefficiency))


global_efficiency = calculateSixCoV(BIOCARD_orig_globalefficiency + VMAP_orig_globalefficiency,
BIOCARD_orig_globalefficiency + VMAP_mean_globalefficiency,
BIOCARD_mean_globalefficiency + VMAP_orig_globalefficiency,
BIOCARD_mean_std_globalefficiency + VMAP_mean_std_globalefficiency,
BIOCARD_orig_globalefficiency + VMAP_CycleGan_globalefficiency,
VMAP_orig_globalefficiency + BIOCARD_CycleGan_globalefficiency)


modularity = calculateSixCoV(BIOCARD_orig_modularity + VMAP_orig_modularity,
BIOCARD_orig_modularity + VMAP_mean_modularity,
BIOCARD_mean_modularity + VMAP_orig_modularity,
BIOCARD_mean_std_modularity + VMAP_mean_std_modularity,
BIOCARD_orig_modularity + VMAP_CycleGan_modularity,
VMAP_orig_modularity + BIOCARD_CycleGan_modularity)

average_betweenness = calculateSixCoV(BIOCARD_orig_averagebetweennesscentrality + VMAP_orig_averagebetweennesscentrality,
BIOCARD_orig_averagebetweennesscentrality + VMAP_mean_averagebetweennesscentrality,
BIOCARD_mean_averagebetweennesscentrality + VMAP_orig_averagebetweennesscentrality,
BIOCARD_mean_std_averagebetweennesscentrality + VMAP_mean_std_averagebetweennesscentrality,
BIOCARD_orig_averagebetweennesscentrality + VMAP_CycleGan_averagebetweennesscentrality,
VMAP_orig_averagebetweennesscentrality + BIOCARD_CycleGan_averagebetweennesscentrality)


plot_barcharts(average_betweenness, modularity, global_efficiency)

"""
print("ABC")
u_test_for_datasets(BIOCARD_orig_averagebetweennesscentrality, VMAP_orig_averagebetweennesscentrality)
u_test_for_datasets(BIOCARD_mean_averagebetweennesscentrality, VMAP_orig_averagebetweennesscentrality)
u_test_for_datasets(BIOCARD_orig_averagebetweennesscentrality, VMAP_mean_averagebetweennesscentrality)
u_test_for_datasets(BIOCARD_mean_std_averagebetweennesscentrality, VMAP_mean_std_averagebetweennesscentrality)
u_test_for_datasets(BIOCARD_CycleGan_averagebetweennesscentrality, VMAP_orig_averagebetweennesscentrality)
u_test_for_datasets(BIOCARD_orig_averagebetweennesscentrality, VMAP_CycleGan_averagebetweennesscentrality)
print("_____________________________")

print("global efficiency")
u_test_for_datasets(BIOCARD_orig_globalefficiency, VMAP_orig_globalefficiency)
u_test_for_datasets(BIOCARD_mean_globalefficiency, VMAP_orig_globalefficiency)
u_test_for_datasets(BIOCARD_orig_globalefficiency, VMAP_mean_globalefficiency)
u_test_for_datasets(BIOCARD_mean_std_globalefficiency, VMAP_mean_std_globalefficiency)
u_test_for_datasets(BIOCARD_CycleGan_globalefficiency, VMAP_orig_globalefficiency)
u_test_for_datasets(BIOCARD_orig_globalefficiency, VMAP_CycleGan_globalefficiency)
print("_____________________________")

print("modularity")
u_test_for_datasets(BIOCARD_orig_modularity, VMAP_orig_modularity)
u_test_for_datasets(BIOCARD_mean_modularity, VMAP_orig_modularity)
u_test_for_datasets(BIOCARD_orig_modularity, VMAP_mean_modularity)
u_test_for_datasets(BIOCARD_mean_std_modularity, VMAP_mean_std_modularity)
u_test_for_datasets(BIOCARD_CycleGan_modularity, VMAP_orig_modularity)
u_test_for_datasets(BIOCARD_orig_modularity, VMAP_CycleGan_modularity)
"""
"""
BIOCARD_orig_averagebetweennesscentrality = collect_graph_age_pair(Combined, BIOCARD_path, "averagebetweennesscentrality", "BIOCARD")
BIOCARD_mean_averagebetweennesscentrality = collect_graph_age_pair(Combined, BIOCARD_MEAN, "averagebetweennesscentrality", "BIOCARD")
BIOCARD_mean_std_averagebetweennesscentrality = collect_graph_age_pair(Combined, BIOCARD_MEAN_ComBat, "averagebetweennesscentrality", "BIOCARD")
BIOCARD_CycleGan_averagebetweennesscentrality = collect_graph_age_pair(Combined, BIOCARD_CycleGAN, "averagebetweennesscentrality", "BIOCARD")
VMAP_orig_averagebetweennesscentrality = collect_graph_age_pair(Combined, VMAP_path, "averagebetweennesscentrality", "VMAP")
VMAP_mean_averagebetweennesscentrality = collect_graph_age_pair(Combined, VMAP_MEAN, "averagebetweennesscentrality", "VMAP")
VMAP_mean_std_averagebetweennesscentrality = collect_graph_age_pair(Combined, VMAP_MEAN_ComBat, "averagebetweennesscentrality", "VMAP")
VMAP_CycleGan_averagebetweennesscentrality = collect_graph_age_pair(Combined, VMAP_CycleGAN, "averagebetweennesscentrality", "VMAP")



# Increase font size
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)  # fontsize of the figure title

# Here, you should have your data in data_array, where each element is a pair of data points (data1, data2)
data_array = [(BIOCARD_orig_averagebetweennesscentrality, VMAP_orig_averagebetweennesscentrality), (BIOCARD_orig_averagebetweennesscentrality, VMAP_mean_averagebetweennesscentrality),  (BIOCARD_mean_averagebetweennesscentrality, VMAP_orig_averagebetweennesscentrality),(BIOCARD_orig_averagebetweennesscentrality, VMAP_CycleGan_averagebetweennesscentrality),  (BIOCARD_CycleGan_averagebetweennesscentrality, VMAP_orig_averagebetweennesscentrality), (BIOCARD_mean_std_averagebetweennesscentrality, VMAP_mean_std_averagebetweennesscentrality)]

fig, axs = plt.subplots(1, 6, figsize=(30, 6), sharey=True) # create a 1x6 grid of plots

for idx, (data1, data2) in enumerate(data_array):
    ax = axs[idx] # find the position in the grid
    plot_scatter(ax, data1, data2, f'Correlation between Age and Global Efficiency {idx+1}')
    if idx != 0: # hide y-axis labels for all but the first subplot
        ax.yaxis.set_visible(False)

plt.tight_layout()
plt.savefig("/nfs2/xuh11/Connectome/Visualize_Mean_Shift/averagebetweennesscentrality_linear.png")
"""
"""
plot_scatter(BIOCARD_orig_modularity, VMAP_orig_modularity)
plot_scatter(BIOCARD_mean_modularity, VMAP_orig_modularity)
plot_scatter(BIOCARD_orig_modularity, VMAP_mean_modularity)
plot_scatter(BIOCARD_mean_std_modularity, VMAP_mean_std_modularity)
plot_scatter(BIOCARD_CycleGan_modularity, VMAP_orig_modularity)
plot_scatter(BIOCARD_orig_modularity, VMAP_CycleGan_modularity)

BIOCARD_orig_globalefficiency = collect_graph_age_pair(Combined, BIOCARD_path, "globalefficiency", "BIOCARD")
BIOCARD_mean_globalefficiency = collect_graph_age_pair(Combined, BIOCARD_MEAN, "globalefficiency", "BIOCARD")
BIOCARD_mean_std_globalefficiency = collect_graph_age_pair(Combined, BIOCARD_MEAN_ComBat, "globalefficiency", "BIOCARD")
BIOCARD_CycleGan_globalefficiency = collect_graph_age_pair(Combined, BIOCARD_CycleGAN, "globalefficiency", "BIOCARD")
VMAP_orig_globalefficiency = collect_graph_age_pair(Combined, VMAP_path, "globalefficiency", "VMAP")
VMAP_mean_globalefficiency = collect_graph_age_pair(Combined, VMAP_MEAN, "globalefficiency", "VMAP")
VMAP_mean_std_globalefficiency = collect_graph_age_pair(Combined, VMAP_MEAN_ComBat, "globalefficiency", "VMAP")
VMAP_CycleGan_globalefficiency = collect_graph_age_pair(Combined, VMAP_CycleGAN, "globalefficiency", "VMAP")

plot_scatter(BIOCARD_orig_globalefficiency, VMAP_orig_globalefficiency)
plot_scatter(BIOCARD_mean_globalefficiency, VMAP_orig_globalefficiency)
plot_scatter(BIOCARD_orig_globalefficiency, VMAP_mean_globalefficiency)
plot_scatter(BIOCARD_mean_std_globalefficiency, VMAP_mean_std_globalefficiency)
plot_scatter(BIOCARD_CycleGan_globalefficiency, VMAP_orig_globalefficiency)
plot_scatter(BIOCARD_orig_globalefficiency, VMAP_CycleGan_globalefficiency)
"""

"""
generate_plots(BIOCARD_orig_assortativity,
BIOCARD_mean_assortativity,
BIOCARD_mean_std_assortativity,
BIOCARD_CycleGan_assortativity,
VMAP_orig_assortativity,
VMAP_mean_assortativity,
VMAP_mean_std_assortativity,
VMAP_CycleGan_assortativity)

generate_plots(BIOCARD_orig_averagebetweennesscentrality,
BIOCARD_mean_averagebetweennesscentrality,
BIOCARD_mean_std_averagebetweennesscentrality,
BIOCARD_CycleGan_averagebetweennesscentrality,
VMAP_orig_averagebetweennesscentrality,
VMAP_mean_averagebetweennesscentrality,
VMAP_mean_std_averagebetweennesscentrality,
VMAP_CycleGan_averagebetweennesscentrality)

generate_plots(BIOCARD_orig_characteristicpathlength,
BIOCARD_mean_characteristicpathlength,
BIOCARD_mean_std_characteristicpathlength,
BIOCARD_CycleGan_characteristicpathlength,
VMAP_orig_characteristicpathlength,
VMAP_mean_characteristicpathlength,
VMAP_mean_std_characteristicpathlength,
VMAP_CycleGan_characteristicpathlength)

generate_plots(BIOCARD_orig_globalefficiency,
BIOCARD_mean_globalefficiency,
BIOCARD_mean_std_globalefficiency,
BIOCARD_CycleGan_globalefficiency,
VMAP_orig_globalefficiency,
VMAP_mean_globalefficiency,
VMAP_mean_std_globalefficiency,
VMAP_CycleGan_globalefficiency)

generate_plots(BIOCARD_orig_modularity,
BIOCARD_mean_modularity,
BIOCARD_mean_std_modularity,
BIOCARD_CycleGan_modularity,
VMAP_orig_modularity,
VMAP_mean_modularity,
VMAP_mean_std_modularity,
VMAP_CycleGan_modularity)
"""