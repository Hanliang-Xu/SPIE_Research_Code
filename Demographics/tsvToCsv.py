import pandas as pd

# Read the TSV file
df = pd.read_csv('/nfs2/xuh11/Connectome/Demographics/VMAP_Demographics_Chenyu.tsv', sep='\t')

# Write the DataFrame to a CSV file
df.to_csv('/nfs2/xuh11/Connectome/Demographics/VMAP_Demographics_Chenyu.csv', index=False)