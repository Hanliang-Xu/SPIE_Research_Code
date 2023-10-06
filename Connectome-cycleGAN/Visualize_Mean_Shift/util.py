import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.stats as stats

BIOCARD_MEAN_ComBat = Path('/nfs2/xuh11/ComBat_BIOCARD')
BIOCARD_CycleGAN = Path('/nfs2/xuh11/Connectome/CycleGan-results/BIOCARD')


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
            if (sub == "sub-JHU666199" and ses == "ses-160503" and input_folder != BIOCARD_MEAN_ComBat and input_folder != BIOCARD_CycleGAN):
                result.append(graph_measure_pipe(folderpath / "run-2", False, graph_measure))
            elif (sub == "sub-JHU989472" and ses == "ses-150427" and input_folder != BIOCARD_MEAN_ComBat and input_folder != BIOCARD_CycleGAN):
                result.append(graph_measure_pipe(folderpath / "run-1", False, graph_measure))
            elif(run.name.startswith('run')):
                result.append(graph_measure_pipe(folderpath, True, graph_measure))
            else:
                result.append(graph_measure_pipe(folderpath, False, graph_measure))
            break
    return result

# Iterate over all rows in finalBIOCARD.csv
def collect_graph_age_pair(demographics, input_folder, graph_measure, dataset):
    result = []
    for _, row in demographics.iterrows():
        sub = row["sub_" + dataset]
        ses = row["ses_" + dataset]
        age = row["age_" + dataset]
        # Construct the path to the number.csv file
        folderpath = input_folder / sub / ses
        for run in folderpath.iterdir():
            if (sub == "sub-JHU666199" and ses == "ses-160503" and input_folder != BIOCARD_MEAN_ComBat and input_folder != BIOCARD_CycleGAN):
                result.append(graph_measure_pipe(folderpath / "run-2", False, graph_measure))
            elif (sub == "sub-JHU989472" and ses == "ses-150427" and input_folder != BIOCARD_MEAN_ComBat and input_folder != BIOCARD_CycleGAN):
                result.append(graph_measure_pipe(folderpath / "run-1", False, graph_measure))
            elif(run.name.startswith('run')):
                result.append(graph_measure_pipe(folderpath, True, graph_measure))
            else:
                result.append(graph_measure_pipe(folderpath, False, graph_measure))
            result.append(age)
            break
    return result

def generate_plots(orig1, mean1, mean_sub1, mean_cycleGAN1, orig2, mean2, mean_sub2, mean_cycleGAN2):
    fig, axs = plt.subplots(nrows=1, ncols=8, figsize=(18, 6), sharey=True)

    # Define two sets of colors
    colors_red = sns.color_palette("Reds_r", 5) # 4 colors, in decreasing order
    colors_green = sns.color_palette("Greens_r", 5) # 4 colors, in decreasing order

    datasets = [orig1, mean1, mean_cycleGAN1, mean_sub1, orig2, mean2, mean_cycleGAN2, mean_sub2]
    colors = [colors_red[1], colors_red[2], colors_red[3], colors_red[4], colors_green[1], colors_green[2], colors_green[3], colors_green[4]]
    titles = ['Original BIOCARD', 'Mean-shifted BIOCARD', 'CycleGAN BIOCARD','Combatted BIOCARD', 'Original VMAP', 'Mean-shifted VMAP', 'CycleGAN VMAP',  'Combatted VMAP']

    # Compute global min, max and range
    global_min = min(min(data) for data in datasets)
    global_max = max(max(data) for data in datasets)
    global_range = global_max - global_min

    # Set global min and max with some padding
    axs[0].set_ylim(global_min - 0.1 * global_range, global_max + 0.2 * global_range)

    # Set mean lines
    average1 = np.mean(orig1)
    average2 = np.mean(orig2)
    average3 = np.mean(mean_sub1)
    mean4 = np.mean(mean1)
    mean5 = np.mean(mean_cycleGAN1)
    mean6 = np.mean(mean2)
    mean7 = np.mean(mean_cycleGAN2)
    mean8 = np.mean(mean_sub2)
    print(mean8 - average3)
    means = [average1, average2, average3, mean4, mean5, mean6, mean7, mean8] # List of mean variables
    for i, mean in enumerate(means, 1): # The second argument 1 in enumerate is for starting index
        print(f'mean{i} = {mean}')
    

    for idx, (ax, data, color, title) in enumerate(zip(axs, datasets, colors, titles)):
        sns.violinplot(ax=ax, y=data, color=color)
        ax.set_title(title)

        # add mean lines for first six plots
        if title in titles[0] or title in titles[5:7]: 
            ax.axhline(y=average1, color=colors_red[0], linestyle='--', linewidth=5.0)
        if title in titles[1:3] or title in titles[4]: 
            ax.axhline(y=average2, color=colors_green[0], linestyle='--', linewidth=5.0)
        if title in titles[3] or title in titles[7]: 
            ax.axhline(y=average3, color="black", linestyle='--', linewidth=5.0)

        ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    plt.savefig('/nfs2/xuh11/Connectome/Visualize_Mean_Shift/ABC.png', dpi=600)




def calculateSixCoV(data1, data2, data3, data4, data5, data6):
    result = []
    result.append(np.std(data1) / np.mean(data1) * 100)
    result.append(np.std(data2) / np.mean(data2) * 100)
    result.append(np.std(data3) / np.mean(data3) * 100)
    result.append(np.std(data4) / np.mean(data4) * 100)
    result.append(np.std(data5) / np.mean(data5) * 100)
    result.append(np.std(data6) / np.mean(data6) * 100)
    return result


def plot_barcharts(average_betweenness, global_efficiency, modularity):
    labels = ["reference", "shifted site 1", "shifted site 2", "neuroCombat", "CycleGAN site 1", "CycleGAN site 2"]
    titles = ["average betweenness centrality", "global efficiency", "modularity"]
    data = [average_betweenness, global_efficiency, modularity]
    colors = ['royalblue', 'seagreen', 'firebrick', 'darkorange', 'mediumpurple', 'dodgerblue']

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    for i, ax in enumerate(axs):
        ax.bar(labels, data[i], color=colors)
        ax.set_title(titles[i], fontsize=16)  # larger title
        for label in ax.get_xticklabels():  # larger xtick labels
            label.set_size(14)
            label.set_rotation(45)
        for label in ax.get_yticklabels():  # larger ytick labels
            label.set_size(14)

    axs[0].set_ylabel('CoV (%)', fontsize=16)  # larger y-label for the first subplot only

    plt.tight_layout()
    plt.savefig('/nfs2/xuh11/Connectome/Visualize_Mean_Shift/box_plot.png', dpi=600)

"""
def plot_scatter(data1, data2):
    # Increase font size
    plt.rc('font', size=18)          # controls default text sizes
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    plt.rc('legend', fontsize=16)    # legend fontsize
    plt.rc('figure', titlesize=22)  # fontsize of the figure title
    
    # The array is split into two arrays, one for x and one for y
    x1 = np.array(data1[::2])
    y1 = np.array(data1[1::2])
    x2 = np.array(data2[::2])
    y2 = np.array(data2[1::2])
    
    # Concatenate x1 and x2, y1 and y2 for combined statistical analysis
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    plt.figure(figsize=(10, 10))
    plt.scatter(y1, x1, color='b', label='BIOCARD')
    plt.scatter(y2, x2, color='r', label='VMAP')

    coefficients = np.polyfit(y, x, 1)
    poly = np.poly1d(coefficients)
    plt.plot(y, poly(y), color='g', label=f'Trend line: {poly}')

    plt.xlabel('Age')
    plt.ylabel('Modularity')
    plt.title('Correlation between Age and Modularity')
    plt.legend()

    # Performing the statistical analysis
    correlation_coef, p_value = stats.pearsonr(y, x)
    print(f'Pearson correlation coefficient: {correlation_coef:.4f}')
    print(f'p-value: {p_value:.4f}')
    
    plt.show()
"""
# Function to plot scatter
def plot_scatter(ax, data1, data2, title):
    x1 = np.array(data1[::2])
    y1 = np.array(data1[1::2])
    x2 = np.array(data2[::2])
    y2 = np.array(data2[1::2])

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    ax.scatter(y1, x1, color='b', label='site 2')
    ax.scatter(y2, x2, color='r', label='site 1')

    coefficients = np.polyfit(y, x, 1)
    poly = np.poly1d(coefficients)
    ax.plot(y, poly(y), color='g', label=f'Trend line: {poly}')

    ax.xaxis.label.set_fontsize(22)  # set x-axis label size
    ax.yaxis.label.set_fontsize(22)  # set y-axis label size

    ax.tick_params(axis='both', labelsize=20)  # set tick label size

    ax.set_xlabel('Age')
    ax.set_ylabel('Modularity')
    ax.set_title(title)
    #ax.legend()

    correlation_coef, p_value = stats.pearsonr(y, x)
    print(f'Pearson correlation coefficient: {correlation_coef:.4f}')
    print(f'p-value: {p_value:.4f}')

def u_test_for_datasets(data1, data2):
    
    # Mann-Whitney U test (for non-normally distributed data)
    u_stat, p_val = stats.mannwhitneyu(data1, data2)

    print(f"U-statistic: {u_stat}")
    print(f"P-value: {p_val}")
    
