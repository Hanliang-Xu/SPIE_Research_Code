import pandas as pd
import numpy as np
import subprocess
# specify csv file name
csv_filename = '/nfs2/xuh11/ConnectomeSpecialOnRAMBAM/testDiaganol/CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv'

# read the csv file (assuming it's a csv with only numerical data)
data = pd.read_csv(csv_filename, header=None)

# convert DataFrame to numpy array
data_np = data.to_numpy()

# make diagonal elements zero
np.fill_diagonal(data_np, 0)

# convert back to DataFrame
data = pd.DataFrame(data_np)

# save the modified data to new csv file
new_csv_filename = '/nfs2/xuh11/ConnectomeSpecialOnRAMBAM/testDiaganol/CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv'
data.to_csv(new_csv_filename, index=False, header=False)

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

matlab_line("/nfs2/xuh11/ConnectomeSpecialOnRAMBAM/testDiaganol")