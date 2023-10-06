import pandas as pd

# Define file path
file_path = "/nfs2/xuh11/Connectome/Demographics/BIOCARD_Demographics_Chenyu.csv"

# Read CSV file into a DataFrame
df = pd.read_csv(file_path)

# Reorder the columns
df = df[[col for col in df.columns if col not in ['sex_BIOCARD', 'diagnosis_BIOCARD', 'age_BIOCARD']] + ['sex_BIOCARD', 'diagnosis_BIOCARD', 'age_BIOCARD']]

# Save the reordered DataFrame back to the CSV file
df.to_csv(file_path, index=False)