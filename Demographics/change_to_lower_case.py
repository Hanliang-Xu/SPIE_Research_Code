import pandas as pd

# Load the CSV file
df = pd.read_csv('/nfs2/xuh11/Connectome/Demographics/BIOCARD_Demographics_Chenyu.csv')
print(df.columns)
# Replace 'NORMAL' with 'Normal' in the 'diagnosis_BIOCARD' column
df['diagnosis_BIOCARD'] = df['diagnosis_BIOCARD'].replace('NORMAL', 'Normal')

# Save the DataFrame back to the same CSV file
df.to_csv('/nfs2/xuh11/Connectome/Demographics/BIOCARD_Demographics_Chenyu.csv', index=False)
