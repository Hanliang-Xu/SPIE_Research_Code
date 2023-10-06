from pathlib import Path
from neuroCombat import neuroCombat
import pandas as pd
import numpy as np
import os

BIOCARD_path = Path('/nfs2/harmonization/raw/BIOCARD_ConnectomeSpecial')
VMAP_path = Path('/nfs2/harmonization/raw/VMAP_ConnectomeSpecial')

both_demo = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/matchVMAPwithBIOCARD(sameSexCog0_9Age).csv")
BIOCARD_demo = both_demo.iloc[:, -5:]
VMAP_demo = both_demo.iloc[:, :5]


def convert_csv_to_upper_half_array(csv_path):
    # Read the csv file
    df = pd.read_csv(csv_path, header=None)

    # Convert the dataframe to a numpy array
    matrix = df.values

    # Get the indices for the upper half of the matrix, excluding the diagonal
    upper_half_indices = np.triu_indices(matrix.shape[0])

    # Get the values of the upper half of the matrix
    upper_half_values = matrix[upper_half_indices]

    # Reshape the array to be 1 by x
    upper_half_values = upper_half_values.reshape(1, -1)

    return upper_half_values

# Iterate over all rows in finalBIOCARD.csv
def collect_graph_measure(demographics, input_folder, type_of_matrix, batch):
    arrays = []
    if (batch == 1):
        dataset = "_VMAP"
    elif (batch == 2):
        dataset = "_BIOCARD"
    else:
        dataset = "NaN"
    
    for _, row in demographics.iterrows():
        sub = row["sub" + dataset]
        ses = row["ses" + dataset]
        sex = row["sex" + dataset]
        age = row["age" + dataset]
        # Construct the path to the number.csv file
        folderpath = input_folder / sub / ses
        for run in folderpath.iterdir():
            if (sub == "sub-JHU666199" and ses == "ses-160503"):
                array = convert_csv_to_upper_half_array(folderpath / "run-2" / type_of_matrix)
            elif (sub == "sub-JHU989472" and ses == "ses-150427"):
                array = convert_csv_to_upper_half_array(folderpath / "run-1" / type_of_matrix)
            elif(run.name.startswith('run')):
                array = (convert_csv_to_upper_half_array(folderpath / "run-1" / type_of_matrix) + convert_csv_to_upper_half_array(folderpath / "run-2" / type_of_matrix)) / 2.0
            else:
                array = convert_csv_to_upper_half_array(folderpath / type_of_matrix)
            array = np.append(array, [batch, sex, age])
            arrays.append(array)
            break
    return pd.DataFrame(arrays)

VMAP_MeanLength = collect_graph_measure(VMAP_demo, VMAP_path, "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv", 1)
BIOCARD_MeanLength = collect_graph_measure(BIOCARD_demo, BIOCARD_path, "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv", 2)
VMAP_NumStreamlines = collect_graph_measure(VMAP_demo, VMAP_path, "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000.csv", 1)
BIOCARD_NumStreamlines = collect_graph_measure(BIOCARD_demo, BIOCARD_path, "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000.csv", 2)
MeanLength = pd.concat([VMAP_MeanLength, BIOCARD_MeanLength])
NumStreamlines = pd.concat([VMAP_NumStreamlines, BIOCARD_NumStreamlines])

data = MeanLength.iloc[:, :-3].transpose()

# Get last three columns
covars = MeanLength.iloc[:, -3:]

# Rename them
covars.columns = ['batch', 'sex', 'age']

categorical_cols = ['sex']
continuous_cols = ['age']
batch_col = 'batch'

np.save('/nfs2/xuh11/Connectome/ComBat/before_combat.npy', data)

data_combat = neuroCombat(dat=data,
    covars=covars,
    batch_col=batch_col,
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols)["data"]

np.save('/nfs2/xuh11/Connectome/ComBat/collect_all.npy', data_combat)

VMAP_batch = data_combat[:, :int(data_combat.shape[1]/2)]
BIOCARD_batch = data_combat[:, int(data_combat.shape[1]/2):]

def convert_back_to_csv(demographics, combat_result, dataset_name, type_of_matrix):
    combat_result = np.transpose(combat_result)

    counter_for_sub = 0
    # Loop over the columns of the array (since the array is now 168x3570)
    for i in range(combat_result.shape[0]):
        # Create an empty 84 by 84 DataFrame
        df = pd.DataFrame(np.zeros((84, 84)))
        # Fill in the upper-right triangle
        counter = 0
        element_counter = 0
        for j in range(84, 0, -1):
            num_elements = j
            df.iloc[counter, counter:counter+j+1] = combat_result[i, element_counter:element_counter+num_elements]
            element_counter += num_elements
            counter += 1

        df = df + df.T

        # convert DataFrame to numpy array
        data_np = df.to_numpy()
        # divide diagonal elements by 2.0
        np.fill_diagonal(data_np, np.diag(data_np) / 2.0)
        # convert back to DataFrame
        df = pd.DataFrame(data_np)

        df[df < 0] = 0
        print(df)
        print(df.shape)
        # Save the DataFrame to a csv file
        sub = demographics.loc[counter_for_sub, 'sub_' + dataset_name]
        ses = demographics.loc[counter_for_sub, 'ses_' + dataset_name]
        counter_for_sub += 1
        output_dir = f"/nfs2/xuh11/ComBat_{dataset_name}/{sub}/{ses}/"
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Then, save the DataFrame to a CSV file
        csv_file = f"{output_dir}{type_of_matrix}"
        df.to_csv(csv_file, header=False, index=False)

convert_back_to_csv(BIOCARD_demo, BIOCARD_batch, "BIOCARD", "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv")
convert_back_to_csv(VMAP_demo, VMAP_batch, "VMAP", "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv")