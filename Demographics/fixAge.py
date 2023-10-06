import pandas as pd

# Load the CSV file
df = pd.read_csv('/nfs2/xuh11/Connectome/Demographics/isItCorrect.csv')

# Initialize count
count = 0
info_list = []

# Iterate over DataFrame rows
for index, row in df.iterrows():
    # Calculate value
    print(row['year'] + row['month'] / 12.0 - row['birthyr'])
    if (row['year'] + row['month'] / 12.0 - row['birthyr'] - row['age_BIOCARD']) >= 1:
        # Increment count
        info_list.append((row['sub_BIOCARD'], row['ses_BIOCARD']))

# Print the list
for info in info_list:
    print(info)