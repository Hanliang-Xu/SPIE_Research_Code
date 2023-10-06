import pandas as pd

# Load the CSV file
df = pd.read_csv('/nfs2/xuh11/Connectome/Demographics/BIOCARD_Demographics.csv')

# Delete the 'delete' column
df = df.drop('delete', axis=1)

# Save the DataFrame back to the same CSV file
df.to_csv('/nfs2/xuh11/Connectome/Demographics/BIOCARD_Demographics.csv', index=False)

