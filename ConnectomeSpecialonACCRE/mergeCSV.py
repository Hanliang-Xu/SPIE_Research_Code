import pandas as pd

# Read the two CSV files
df1 = pd.read_csv('/nfs2/xuh11/Connectome/BLSA_list_freesurfer.csv')
df2 = pd.read_csv('/nfs2/xuh11/Connectome/BLSA_list.csv')

# Find the common rows
common = pd.merge(df1, df2, how='inner')

# Write the common rows to a new CSV file
common.to_csv('final_BLSA_list.csv', index=False)