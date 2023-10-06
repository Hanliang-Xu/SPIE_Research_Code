import os
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

# find ./ -type f -name GraphMeasure* -exec rm -f {} ";"
error = []

BIOCARD_assortativity = []
BIOCARD_averagebetweennesscentrality = []
BIOCARD_characteristicpathlength = []
BIOCARD_globalefficiency = []
BIOCARD_modularity = []
VMAP_assortativity = []
VMAP_averagebetweennesscentrality = []
VMAP_characteristicpathlength = []
VMAP_globalefficiency = []
VMAP_modularity = []

BIOCARD_path = Path('/nfs2/harmonization/raw/BIOCARD_ConnectomeSpecial/')
VMAP_path = Path('/nfs2/harmonization/raw/VMAP_ConnectomeSpecial/')
BLSA_path = Path('/nfs2/harmonization/raw/BLSA_ConnectomeSpecial')
BIOCARD_processed = Path('/nfs2/xuh11/Mean_Shift/Mean_Shift_BIOCARD')
VMAP_processed = Path('/nfs2/xuh11/Mean_Shift/Mean_Shift_VMAP')

# Load the CSV file
BIOCARD_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalBIOCARD.csv")
VMAP_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalVMAP.csv")
BLSA_df = pd.read_csv("/nfs2/xuh11/Connectome/ConnectomeSpecialonACCRE/generate_graph_BLSA.csv")

def readCSVfile(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        first_line = next(reader)  # Get the first line
        first_number = float(first_line[0])  # Convert the first element to a float
    return first_number

def matlab_line(folder):
    print(folder)
    command = f"""
    cd /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/support_scripts &&
    export COMMAND="calculategms('{folder}','{folder}');exit" &&
    echo ${{COMMAND}} &&
    matlab -nodisplay -nojvm -nosplash -nodesktop -r ${{COMMAND}}
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    process.communicate()

def graph_measure_pipe(path, starts_with_run, assortativity, averagebetweennesscentrality, characteristicpathlength, globalefficiency, modularity) :
    OUTPUTDIR = path
    if os.path.isfile(OUTPUTDIR / "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv") or os.path.isfile(OUTPUTDIR / "run-1" / "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv"):
        
        if (starts_with_run) :
            #matlab_line(path / "run-1")
            #matlab_line(path / "run-2")
            assortativity.append((readCSVfile(path / "run-1" / "GraphMeasure_assortativity.csv") + readCSVfile(path / "run-2" / "GraphMeasure_assortativity.csv")) / 2)
            averagebetweennesscentrality.append((readCSVfile(path / "run-1" / "GraphMeasure_averagebetweennesscentrality.csv") + readCSVfile(path / "run-2" / "GraphMeasure_averagebetweennesscentrality.csv")) / 2)
            characteristicpathlength.append((readCSVfile(path / "run-1" / "GraphMeasure_characteristicpathlength.csv") + readCSVfile(path / "run-2" / "GraphMeasure_characteristicpathlength.csv")) / 2)
            globalefficiency.append((readCSVfile(path / "run-1" / "GraphMeasure_globalefficiency.csv") + readCSVfile(path / "run-2" / "GraphMeasure_globalefficiency.csv")) / 2)
            modularity.append((readCSVfile(path / "run-1" / "GraphMeasure_modularity.csv") + readCSVfile(path / "run-2" / "GraphMeasure_modularity.csv")) / 2)
        else:
            #matlab_line(path)
            assortativity.append(readCSVfile(path / "GraphMeasure_assortativity.csv"))
            averagebetweennesscentrality.append(readCSVfile(path / "GraphMeasure_averagebetweennesscentrality.csv"))
            characteristicpathlength.append(readCSVfile(path / "GraphMeasure_characteristicpathlength.csv"))
            globalefficiency.append(readCSVfile(path / "GraphMeasure_globalefficiency.csv"))
            modularity.append(readCSVfile(path / "GraphMeasure_modularity.csv"))
    else:
        error.append(path)

# Iterate over all rows in finalBIOCARD.csv
def collect_graph_measure(demographics, input_folder, assortativity, averagebetweennesscentrality, characteristicpathlength, globalefficiency, modularity):
    for _, row in demographics.iterrows():
        sub = row["sub"]
        ses = row["ses"]
        # Construct the path to the number.csv file
        folderpath = input_folder / sub / ses
        for run in folderpath.iterdir():
            if ((sub == "sub-JHU308071" and ses == "ses-150811") or (sub == "sub-JHU666199" and ses == "ses-160503")):
                graph_measure_pipe(folderpath / "run-2", False, assortativity, averagebetweennesscentrality, characteristicpathlength, globalefficiency, modularity)
            elif(run.name.startswith('run')):
                graph_measure_pipe(folderpath, True, assortativity, averagebetweennesscentrality, characteristicpathlength, globalefficiency, modularity)
            else:
                graph_measure_pipe(folderpath, False, assortativity, averagebetweennesscentrality, characteristicpathlength, globalefficiency, modularity)
            break

collect_graph_measure(VMAP_df, VMAP_path, VMAP_assortativity, VMAP_averagebetweennesscentrality, VMAP_characteristicpathlength, VMAP_globalefficiency, VMAP_modularity)
collect_graph_measure(BIOCARD_df, BIOCARD_path, BIOCARD_assortativity, BIOCARD_averagebetweennesscentrality, BIOCARD_characteristicpathlength, BIOCARD_globalefficiency, BIOCARD_modularity)


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


#print(BIOCARD_assortativity)
#print(VMAP_assortativity)
plot(BIOCARD_assortativity, "BIOCARD", VMAP_assortativity, "VMAP")
plot(BIOCARD_averagebetweennesscentrality, "BIOCARD", VMAP_averagebetweennesscentrality, "VMAP")
plot(BIOCARD_characteristicpathlength, "BIOCARD", VMAP_characteristicpathlength, "VMAP")
plot(BIOCARD_globalefficiency, "BIOCARD", VMAP_globalefficiency, "VMAP")
plot(BIOCARD_modularity, "BIOCARD", VMAP_modularity, "VMAP")