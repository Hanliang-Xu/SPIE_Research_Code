# This script visualizes diffusion tractograms from six directions
# 
# Author: Hanliang Xu
# Date: July 3, 2023

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

BIOCARD_path = Path('/nfs2/harmonization/raw/BIOCARD_ConnectomeSpecial')
VMAP_path = Path('/nfs2/harmonization/raw/VMAP_ConnectomeSpecial')
BIOCARD_processed = Path('/nfs2/xuh11/Mean_Shift/Mean_Shift_BIOCARD')
VMAP_processed = Path('/nfs2/xuh11/Mean_Shift/Mean_Shift_VMAP')
BIOCARD_CycleGAN = Path('/nfs2/xuh11/Connectome/CycleGan-results/BIOCARD')
VMAP_CycleGAN = Path('/nfs2/xuh11/Connectome/CycleGan-results/VMAP')

# Load the CSV file
BIOCARD_sub_ses_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalBIOCARD.csv")
VMAP_sub_ses_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalVMAP.csv")
BIOCARD_CycleGAN_sub = pd.read_csv('/nfs2/xuh11/Connectome/CycleGAN-ipynb/pytorch-CycleGAN-and-pix2pix/datasets/test_set_A.csv')
VMAP_CycleGAN_sub = pd.read_csv('/nfs2/xuh11/Connectome/CycleGAN-ipynb/pytorch-CycleGAN-and-pix2pix/datasets/test_set_B.csv')

average_BIOCARD = []
average_VMAP = []
std = []

def readCSVfile(filepath):
    with open(filepath, "r") as file:
        return float(file.read())

def countZeros(filepath):
    # Read CSV file into a DataFrame
    df = pd.read_csv(filepath, header=None)

    # Convert DataFrame to numpy array
    matrix = df.values

    # Create a new matrix where zeros in the original matrix are 1, and all other values are 0
    new_matrix = np.where(matrix == 0, 1, 0)

    return new_matrix

def iterate_over_all(demo_input, output_folder, metric_to_calculate, type_of_connectomics, number_to_average):
    count = 0
    if (metric_to_calculate == "MEAN"):
        sum = np.zeros((number_to_average))
    elif (metric_to_calculate == "STD_MATRIX"):
        all_matrices = []
    else:
        sum = np.zeros((84, 84))
    
    if (type_of_connectomics == "MEANLENGTH"):
        name_of_output = "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv"
    elif (type_of_connectomics == "NUMSTREAMLINES"):
        name_of_output = "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000.csv"
    else:
        name_of_output = "N/A"
    
    for _, row in demo_input.iterrows():
        sub = row["sub"]
        ses = row["ses"]

        # Construct the path to the number.csv file
        folderpath = output_folder / sub / ses

        for run in folderpath.iterdir():
            if (run.name.startswith('run')):
                filepath1 = folderpath / "run-1" / name_of_output
                filepath2 = folderpath / "run-2" / name_of_output
                # filepath1 is empty becau
                if ((sub == "sub-JHU308071" and ses == "ses-150811") or (sub == "sub-JHU666199" and ses == "ses-160503")):
                    if (metric_to_calculate == "MEAN_MATRIX"):
                        sum += (np.genfromtxt(filepath2, delimiter=','))
                        count += 1
                    elif (metric_to_calculate == "MEAN"):
                        average_BIOCARD.append(pd.read_csv(filepath2).values.mean())
                    elif (metric_to_calculate == "NumOfZeros"):
                        sum += (countZeros(filepath2))
                    elif (metric_to_calculate == "STD_MATRIX"):
                        all_matrices.append(pd.read_csv(filepath2, header=None).values)
                elif (sub == "sub-JHU989472" and ses == "ses-150427"):
                    if (metric_to_calculate == "MEAN_MATRIX"):
                        sum += (np.genfromtxt(filepath1, delimiter=','))
                        count += 1
                    elif (metric_to_calculate == "STD_MATRIX"):
                        all_matrices.append(pd.read_csv(filepath1, header=None).values)
                else:
                    if ((sub == "sub-JHU423254" and ses == "ses-190102")):
                        filepath2 = "/nfs2/xuh11/notWorkingImages/sub-JHU423254_ses-190102_run-2/" + name_of_output
                    elif ((sub == "sub-JHU830375" and ses == "ses-171212")):
                        filepath1 = "/nfs2/xuh11/notWorkingImages/sub-JHU830375_ses-171212_run-1/" + name_of_output
                    if (metric_to_calculate == "MEAN_MATRIX"):
                        count += 1
                        sum += ((np.genfromtxt(filepath1, delimiter=',') + np.genfromtxt(filepath2, delimiter=',')) / 2.0 )
                    elif (metric_to_calculate == "MEAN"):
                        average_BIOCARD.append((pd.read_csv(filepath1).values.mean() + pd.read_csv(filepath2).values.mean()) / 2)
                    elif (metric_to_calculate == "NumOfZeros"):
                        sum += ((countZeros(filepath1) + (countZeros(filepath2))) / 2.0)
                    elif (metric_to_calculate == "STD_MATRIX"):
                        all_matrices.append((pd.read_csv(filepath1, header=None).values + pd.read_csv(filepath2, header=None).values) / 2)

            else:
                filepath = folderpath / name_of_output
                if (metric_to_calculate == "MEAN_MATRIX"):
                    count += 1
                    sum += (np.genfromtxt(filepath, delimiter=','))
                elif (metric_to_calculate == "MEAN"):
                    if (output_folder == BIOCARD_path):
                        average_BIOCARD.append(pd.read_csv(filepath).values.mean())
                    else:
                        if (sub == "sub-094"):
                            filepath = "/nfs2/xuh11/notWorkingImages/temp-sub-094_ses-EPOCH1x210681/" + name_of_output
                        average_VMAP.append(pd.read_csv(filepath).values.mean())
                        if (pd.read_csv(filepath).values.mean() < 1700):
                            print(filepath)
                            print(pd.read_csv(filepath).values.mean())
                elif (metric_to_calculate == "NumOfZeros"):
                    sum += (countZeros(filepath))
                elif (metric_to_calculate == "STD_MATRIX"):
                    all_matrices.append(pd.read_csv(filepath, header=None).values)
            break
    if (metric_to_calculate == "STD_MATRIX"):
        data = np.array(all_matrices)
        return np.std(data, axis=0)
    print(count)
    return sum / number_to_average

    # Now write the collected numbers to a new CSV file
    #with open("/nfs2/xuh11/Connectome/Analysis/BIOCARD_NumStreamlines.csv", "w", newline="") as file:
    #    writer = csv.writer(file)
    #    for number in BIOCARD_MEANLENGTH_MEAN:
    #        writer.writerow([number])  # write each number on a new line


# Prepare to collect all numbers
BIOCARD_MEANLENGTH_MEAN_CycleGAN = iterate_over_all(BIOCARD_CycleGAN_sub, BIOCARD_CycleGAN, "MEAN_MATRIX", "MEANLENGTH", 16.0)
BIOCARD_MEANLENGTH_MEAN = iterate_over_all(BIOCARD_sub_ses_df, BIOCARD_path, "MEAN_MATRIX", "MEANLENGTH", 84.0)
BIOCARD_NUMSTREAMLINES_MEAN_CycleGAN = iterate_over_all(BIOCARD_CycleGAN_sub, BIOCARD_CycleGAN, "MEAN_MATRIX", "NUMSTREAMLINES", 16.0)
BIOCARD_NUMSTREAMLINES_MEAN = iterate_over_all(BIOCARD_sub_ses_df, BIOCARD_path, "MEAN_MATRIX", "NUMSTREAMLINES", 84.0)

#BIOCARD_NUMSTREAMLINES_MEAN = iterate_over_all(BIOCARD_sub_ses_df, BIOCARD_path, "MEAN_MATRIX", "NUMSTREAMLINES", 134.0)
VMAP_MEANLENGTH_MEAN_CycleGAN = iterate_over_all(VMAP_CycleGAN_sub, VMAP_CycleGAN, "MEAN_MATRIX", "MEANLENGTH", 16.0)
VMAP_MEANLENGTH_MEAN = iterate_over_all(VMAP_sub_ses_df, VMAP_path, "MEAN_MATRIX", "MEANLENGTH", 84.0)
VMAP_NUMSTREAMLINES_MEAN_CycleGAN = iterate_over_all(VMAP_CycleGAN_sub, VMAP_CycleGAN, "MEAN_MATRIX", "NUMSTREAMLINES", 16.0)
VMAP_NUMSTREAMLINES_MEAN = iterate_over_all(VMAP_sub_ses_df, VMAP_path, "MEAN_MATRIX", "NUMSTREAMLINES", 84.0)
#VMAP_NUMSTREAMLINES_MEAN = iterate_over_all(VMAP_sub_ses_df, VMAP_path, "MEAN_MATRIX", "NUMSTREAMLINES", 134.0)

# Collect Means
#iterate_over_all(BIOCARD_sub_ses_df, BIOCARD_path, "MEAN", "MEANLENGTH", 134)
#iterate_over_all(BIOCARD_sub_ses_df, BIOCARD_path, "MEAN", "NUMSTREAMLINES", 134)
#iterate_over_all(VMAP_sub_ses_df, VMAP_path, "MEAN", "MEANLENGTH", 134)
#iterate_over_all(VMAP_sub_ses_df, VMAP_path, "MEAN", "NUMSTREAMLINES", 134)
#print(sum(average_BIOCARD) / len(average_BIOCARD))
#print(sum(average_VMAP) / len(average_VMAP))


#BIOCARD_MEANLENGTH_STD = iterate_over_all(BIOCARD_sub_ses_df, BIOCARD_path, "STD_MATRIX", "MEANLENGTH", 134.0)
#BIOCARD_NUMSTREAMLINES_STD = iterate_over_all(BIOCARD_sub_ses_df, BIOCARD_path, "STD_MATRIX", "NUMSTREAMLINES", 134.0)
#VMAP_MEANLENGTH_STD = iterate_over_all(VMAP_sub_ses_df, VMAP_path, "STD_MATRIX", "MEANLENGTH", 134.0)
#VMAP_NUMSTREAMLINES_STD = iterate_over_all(VMAP_sub_ses_df, VMAP_path, "STD_MATRIX", "NUMSTREAMLINES", 134.0)

#np.savetxt("/nfs2/xuh11/Connectome/Analysis/BIOCARD_mean_length_std.csv", BIOCARD_MEANLENGTH_STD, delimiter=',')
#np.savetxt("/nfs2/xuh11/Connectome/Analysis/VMAP_mean_length_std.csv", VMAP_MEANLENGTH_STD, delimiter=',')
#np.savetxt("/nfs2/xuh11/Connectome/Analysis/BIOCARD_num_streamlines_std.csv", BIOCARD_NUMSTREAMLINES_STD, delimiter=',')
#np.savetxt("/nfs2/xuh11/Connectome/Analysis/VMAP_num_streamlines_std.csv", VMAP_NUMSTREAMLINES_STD, delimiter=',')


#np.savetxt("/nfs2/xuh11/Connectome/Analysis/BIOCARD_mean_length_mean.csv", BIOCARD_MEANLENGTH_MEAN, delimiter=',')
#np.save("/nfs2/xuh11/Connectome/Analysis/BIOCARD_mean_length_mean.npy", BIOCARD_MEANLENGTH_MEAN)
#np.savetxt("/nfs2/xuh11/Connectome/Analysis/VMAP_mean_length_mean.csv", VMAP_MEANLENGTH_MEAN, delimiter=',')
#np.save("/nfs2/xuh11/Connectome/Analysis/VMAP_mean_length_mean.npy", VMAP_MEANLENGTH_MEAN)
#np.savetxt("/nfs2/xuh11/Connectome/Analysis/BIOCARD_num_streamlines_mean.csv", BIOCARD_NUMSTREAMLINES_MEAN, delimiter=',')
#np.save("/nfs2/xuh11/Connectome/Analysis/BIOCARD_num_streamlines_mean.npy", BIOCARD_NUMSTREAMLINES_MEAN)
#np.savetxt("/nfs2/xuh11/Connectome/Analysis/VMAP_num_streamlines_mean.csv", VMAP_NUMSTREAMLINES_MEAN, delimiter=',')
#np.save("/nfs2/xuh11/Connectome/Analysis/VMAP_num_streamlines_mean.npy", VMAP_NUMSTREAMLINES_MEAN)

def plot(df1, df1_name, df2, df2_name) :
    print("the average of", df1_name, "is", np.nanmean(df1))
    print("the average of", df2_name, "is", np.nanmean(df2))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    sns.boxplot(data=[df1, df2])
    plt.xticks(range(2), [df1_name, df2_name])
    boxplot_title = "Boxplot of " + df1_name + " and " + df2_name
    plt.title(boxplot_title)
    plt.xlabel('Label')
    plt.ylabel('Values')

    # To create histograms
    plt.subplot(1, 2, 2)
    plt.hist(df1, alpha=0.5, label=df1_name, bins=10)
    plt.hist(df2, alpha=0.5, label=df2_name, bins=10)
    plt.legend(loc='upper right')
    boxplot_title = "Histogram of " + df1_name + " and " + df2_name
    plt.title(boxplot_title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

#plot(average_BIOCARD, "BIOCARD", average_VMAP, "VMAP")

"""
# Find indices in an average matrix which are of higher values
indices = np.where((BIOCARD_NUMSTREAMLINES_MEAN - VMAP_NUMSTREAMLINES_MEAN) > 10000)
# Print the indices
for index in zip(indices[0], indices[1]):
    print(index)
"""

import matplotlib.pyplot as plt

def visualize_matrix(array1, array1_name, array2, array2_name, standardize, vmin=0, vmax=0):
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    if (standardize):
        im1 = axes[0].imshow(array1, cmap='viridis', vmin=vmin, vmax=vmax)
        im2 = axes[1].imshow(array2, cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        im1 = axes[0].imshow(array1, cmap='viridis')
        im2 = axes[1].imshow(array2, cmap='viridis')
    
    # Plot arrays in the subplots
    axes[0].set_title(array1_name, fontsize=22)  # Increase title font size
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.ax.tick_params(labelsize=18)  # Increase colorbar font size
    axes[0].tick_params(axis='both', which='major', labelsize=18)  # Increase x and y tick font size
    
    axes[1].set_title(array2_name, fontsize=22)  # Increase title font size
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.ax.tick_params(labelsize=18)  # Increase colorbar font size
    axes[1].tick_params(axis='both', which='major', labelsize=18)  # Increase x and y tick font size

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.savefig("/nfs2/xuh11/Connectome/Visualize_Mean_Shift/difference_matrix.png")


def visualize_matrix_log(array1, array1_name, array2, array2_name) :
    # Create a new figure with 2 subplots side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Create a heatmap for array1 in the first subplot
    im1 = ax[0].imshow(array1, norm=LogNorm(vmin=10**-1.5, vmax=10**4.5), aspect='auto')
    ax[0].set_title(array1_name)
    fig.colorbar(im1, ax=ax[0])

    # Create a heatmap for array2 in the second subplot
    im2 = ax[1].imshow(array2, norm=LogNorm(vmin=10**-1.5, vmax=10**4.5), aspect='auto')
    ax[1].set_title(array2_name)
    fig.colorbar(im2, ax=ax[1])

    # Show the figure
    plt.tight_layout()
    plt.savefig("/nfs2/xuh11/Connectome/Visualize_Mean_Shift/mean_BIOCARD.png")

#BIOCARD_MEANLENGTH_MEAN_CycleGAN = iterate_over_all(BIOCARD_CycleGAN_sub, BIOCARD_CycleGAN, "MEAN_MATRIX", "MEANLENGTH", 16.0)
BIOCARD_MEANLENGTH_MEAN = iterate_over_all(BIOCARD_sub_ses_df, BIOCARD_path, "MEAN_MATRIX", "MEANLENGTH", 84.0)
#BIOCARD_NUMSTREAMLINES_MEAN_CycleGAN = iterate_over_all(BIOCARD_CycleGAN_sub, BIOCARD_CycleGAN, "MEAN_MATRIX", "NUMSTREAMLINES", 16.0)
#BIOCARD_NUMSTREAMLINES_MEAN = iterate_over_all(BIOCARD_sub_ses_df, BIOCARD_path, "MEAN_MATRIX", "NUMSTREAMLINES", 84.0)

#BIOCARD_NUMSTREAMLINES_MEAN = iterate_over_all(BIOCARD_sub_ses_df, BIOCARD_path, "MEAN_MATRIX", "NUMSTREAMLINES", 134.0)
#VMAP_MEANLENGTH_MEAN_CycleGAN = iterate_over_all(VMAP_CycleGAN_sub, VMAP_CycleGAN, "MEAN_MATRIX", "MEANLENGTH", 16.0)
VMAP_MEANLENGTH_MEAN = iterate_over_all(VMAP_sub_ses_df, VMAP_path, "MEAN_MATRIX", "MEANLENGTH", 84.0)
#VMAP_NUMSTREAMLINES_MEAN_CycleGAN = iterate_over_all(VMAP_CycleGAN_sub, VMAP_CycleGAN, "MEAN_MATRIX", "NUMSTREAMLINES", 16.0)
#MAP_NUMSTREAMLINES_MEAN = iterate_over_all(VMAP_sub_ses_df, VMAP_path, "MEAN_MATRIX", "NUMSTREAMLINES", 84.0)

#visualize_matrix_log(BIOCARD_NUMSTREAMLINES_MEAN, "BIOCARD Num Streamlines", VMAP_NUMSTREAMLINES_MEAN, "VMAP Num Streamlines")
#visualize_matrix(BIOCARD_MEANLENGTH_MEAN, "BIOCARD", VMAP_MEANLENGTH_MEAN, "VMAP", True, 0, 180)
visualize_matrix(VMAP_MEANLENGTH_MEAN - BIOCARD_MEANLENGTH_MEAN, "VMAP - BIOCARD", VMAP_MEANLENGTH_MEAN_CycleGAN, "VM", False)
#visualize_matrix(BIOCARD_MEANLENGTH_MEAN_CycleGAN, "BIOCARD Mean Length - CycleGAN", VMAP_MEANLENGTH_MEAN, "VMAP Mean Length - original", True, 0, 200)
#visualize_matrix_log(BIOCARD_NUMSTREAMLINES_MEAN, "BIOCARD Num Streamlines - original", VMAP_NUMSTREAMLINES_MEAN_CycleGAN, "VMAP Num Streamlines - CycleGAN")
#visualize_matrix_log(BIOCARD_NUMSTREAMLINES_MEAN_CycleGAN, "BIOCARD Num Streamlines - CycleGAN", VMAP_NUMSTREAMLINES_MEAN, "VMAP Num Streamlines - original")


"""
# Visualize the differences of the average matrices
# Compute the difference between the two matrices
diff_mean_length = VMAP_MEANLENGTH_STD - BIOCARD_MEANLENGTH_STD
# Create a new figure
plt.figure(figsize=(8, 6))
# Create a heatmap for the difference matrix
im = plt.imshow(diff_mean_length, aspect='auto')
# Add a colorbar
plt.colorbar(im)
# Set the title
plt.title("Difference of Mean Length between VMAP and BIOCARD")
# Show the figure
plt.tight_layout()
plt.show()

# Compute the difference between the two matrices
diff_num_streamlines = VMAP_NUMSTREAMLINES_STD - BIOCARD_NUMSTREAMLINES_STD
# Create a new figure
plt.figure(figsize=(8, 6))
# Create a heatmap for the difference matrix
im = plt.imshow(diff_num_streamlines, norm=LogNorm(), aspect='auto')
# Add a colorbar
plt.colorbar(im)
# Set the title
plt.title("Positive Logarithmic Difference between VMAP and BIOCARD Number of Streamlines")
# Show the figure
plt.tight_layout()
plt.show()

# Compute the difference between the two matrices
diff_num_streamlines = BIOCARD_NUMSTREAMLINES_STD - VMAP_NUMSTREAMLINES_STD
# Create a new figure
plt.figure(figsize=(8, 6))
# Create a heatmap for the difference matrix
im = plt.imshow(diff_num_streamlines, norm=LogNorm(), aspect='auto')
# Add a colorbar
plt.colorbar(im)
# Set the title
plt.title("Negative Logarithmic Difference between VMAP and BIOCARD Number of Streamlines")
# Show the figure
plt.tight_layout()
plt.show()
"""