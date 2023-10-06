import pandas as pd
from pathlib import Path

BIOCARD_dataset_path="/Users/leonslaptop/Desktop/SPIE/BIOCARD_sub_ses.csv"
BIOCARD_demographics_path="/Users/leonslaptop/Desktop/SPIE/BIOCARD_demographics.csv"
BIOCARD_diagnosis_path="/Users/leonslaptop/Desktop/SPIE/BIOCARD_diagnosis.csv"

BIOCARD_dataset=pd.read_csv(BIOCARD_dataset_path)
BIOCARD_demographics=pd.read_csv(BIOCARD_demographics_path)
BIOCARD_diagnosis=pd.read_csv(BIOCARD_diagnosis_path)

# drop the column named count
# BIOCARD_dataset=BIOCARD_dataset.drop('count', axis=1)


# Add a column in BIOCARD_sub_ses which records whether we have demographics for the data or not
# BIOCARD_dataset=BIOCARD_dataset.drop('MOFROMBL', axis=1)

df = pd.DataFrame(0, index=range(BIOCARD_dataset.shape[0]), columns=range(1))
BIOCARD_dataset['SEX'] = df

"""
for i in range(BIOCARD_diagnosis.shape[0]):
  BIOCARD_dataset['AGE'].iloc[i] = BIOCARD_dataset['STARTYEAR'].iloc[i] - BIOCARD_dataset['BIRTHYR'].iloc[i] + BIOCARD_dataset['MOFROMBL'].iloc[i] / 12.0
"""
for i in range(BIOCARD_dataset.shape[0]):
  for j in range(BIOCARD_demographics.shape[0]):
    if (BIOCARD_dataset['SUB'].iloc[i][-9:] == BIOCARD_demographics['JHUANONID'].iloc[j]):
      BIOCARD_dataset['SEX'].iloc[i] = BIOCARD_demographics['SEX'].iloc[j]


# Script to count the number of occurance of BIOCARD dataset sub and ses in diagnosis sub and ses
"""
def searchInCsv(csv, startYear, column, id, row):
  if (str(startYear) == ("20" + csv[column].iloc[row][4:6]) and (id == csv["sub_name"].iloc[i][-9:])):
    return True
  return False

for i in range(BIOCARD_diagnosis.shape[0]):
  startYear = BIOCARD_diagnosis["STARTYEAR"].iloc[i]
  sub_name = BIOCARD_diagnosis["JHUANONID"].iloc[i]
  for j in range(BIOCARD_dataset.shape[0]):
    # if the data matches both the ses and the sub:
    dataset_sub_name=BIOCARD_dataset["sub_name"].iloc[j][-9:]
    dataset_startYear_name="20"+BIOCARD_dataset["ses_name"].iloc[j][4:6]
    if (dataset_startYear_name==str(startYear) and dataset_sub_name==sub_name):
      BIOCARD_dataset["count"].iloc[j] += 1
"""


"""
i = 0
index_to_be_removed = []
while (i < BIOCARD_diagnosis.shape[0]):
  if (BIOCARD_diagnosis["have_this_data"].iloc[i] == True):
    index_to_be_removed.append(i)
  i += 1


BIOCARD_diagnosis = BIOCARD_diagnosis.drop(index_to_be_removed)
"""

print(BIOCARD_dataset)

BIOCARD_dataset.to_csv('/Users/leonslaptop/Desktop/SPIE/BIOCARD_sub_ses.csv', index=False)