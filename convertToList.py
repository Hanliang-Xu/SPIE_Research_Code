import pandas as pd
import pickle

# Define file path
file_path = "/nfs2/xuh11/Connectome/missing_CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv.csv"

# Read CSV file into a DataFrame
df = pd.read_csv(file_path)

# Initialize an empty list to store the modified strings
modified_strings = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Get the string from the 'Missing_Directories' column
    string = row['Missing_Directories']
    
    # Convert the last 29 characters except for the 13th character to "_"
    modified_string = ''.join(['_' if i == 13 else char for i, char in enumerate(string[-29:], start=1)])
    
    # Append the modified string to the list
    modified_strings.append(modified_string)

# Define output file path
output_file_path = "/nfs2/xuh11/Connectome/modified_strings.list"

# Open the output file in write mode and save the list
with open(output_file_path, 'w') as f:
    for item in modified_strings:
        f.write(f"{item}\n")