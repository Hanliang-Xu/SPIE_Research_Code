import os
import pandas as pd
from pathlib import Path
import subprocess

# find ./ -type f -name GraphMeasure* -exec rm -f {} ";"
error = []

BIOCARD_processed = Path("/nfs2/xuh11/Mean_Shift_BIOCARD_Std")
BIOCARD_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalBIOCARD.csv")

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

def graph_measure_pipe(path, starts_with_run) :
    OUTPUTDIR = path
    if os.path.isfile(OUTPUTDIR / "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv") or os.path.isfile(OUTPUTDIR / "run-1" / "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv"):
        if (starts_with_run) :
            matlab_line(path / "run-1")
            matlab_line(path / "run-2")
        else:
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
        for run in folderpath.iterdir():
            if (sub == "sub-JHU666199" and ses == "ses-160503"):
                graph_measure_pipe(folderpath / "run-2", False)
            elif (sub == "sub-JHU989472" and ses == "ses-150427"):
                graph_measure_pipe(folderpath / "run-1", False)
            elif(run.name.startswith('run')):
                graph_measure_pipe(folderpath, True)
            else:
                graph_measure_pipe(folderpath, False)
            break

collect_graph_measure(BIOCARD_df, BIOCARD_processed)