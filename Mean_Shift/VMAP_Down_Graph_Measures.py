import os
import pandas as pd
from pathlib import Path
import subprocess

# find ./ -type f -name GraphMeasure* -exec rm -f {} ";"
error = []

VMAP_processed = Path("/nfs2/xuh11/Mean_Shift_VMAP_Down")
VMAP_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalVMAP.csv")

def readCSVfile(filepath):
    with open(filepath, "r") as file:
        return float(file.read())

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

def graph_measure_pipe(path) :
    if os.path.isfile(path / "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv"):
        matlab_line(path)
    else:
        error.append(path)

# Iterate over all rows in finalBIOCARD.csv
def collect_graph_measure(demographics, input_folder):
    for _, row in demographics.iterrows():
        sub = row["sub"]
        ses = row["ses"]
        # Construct the path to the number.csv file
        folderpath = input_folder / sub / ses
        graph_measure_pipe(folderpath)

collect_graph_measure(VMAP_df, VMAP_processed)