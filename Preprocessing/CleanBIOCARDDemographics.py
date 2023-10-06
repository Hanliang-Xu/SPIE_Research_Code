import pandas as pd
from pathlib import Path
from tqdm import tqdm

BIOCARD_Demographics_csv = "/nfs2/xuh11/Connectome/BIOCARD_Demographics/BIOCARD_Demographics_Limited_Data_2022.05.10.csv"
BIOCARD_Diagnosis_csv = "/nfs2/xuh11/Connectome/BIOCARD_Demographics/BIOCARD_DiagnosisData_Limited_2022.05.14.csv"

BIOCARD_Demographics = pd.read_csv(BIOCARD_Demographics_csv)
BIOCARD_Diagnosis = pd.read_csv(BIOCARD_Diagnosis_csv)

data_folder = Path("/nfs2/xuh11/Connectome/QA_BIOCARD")
# Clean demographics information so that the spreedsheet only keeps the images we have
list_subs = [sub.name for sub in data_folder.iterdir() if sub.name.startswith('sub')]
list_not_in_demographics = []
list_not_in_data = []

df = pd.DataFrame(False, index=range(BIOCARD_Demographics.shape[0]), columns=range(1))
BIOCARD_Demographics['have_this_data'] = df

for sub in tqdm(list_subs, total=len(list_subs)):
    for i in range(BIOCARD_Demographics.shape[0]):
        if (BIOCARD_Demographics.loc[i, 'JHUANONID'] == sub):
            BIOCARD_Demographics.loc[i, 'have_this_data'] = True


print(BIOCARD_Demographics)
#print(BIOCARD_Diagnosis)