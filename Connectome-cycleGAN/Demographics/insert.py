import pandas as pd

# Load the CSV file
df = pd.read_csv('/nfs2/xuh11/Connectome/Demographics/VMAP_first_Epoch.csv')

# Make sure the column exists in your dataframe
if 'ses_VMAP' in df.columns:
  # Store the series temporarily
  ses_VMAP_series = df['ses_VMAP']

  # Drop the column from dataframe
  df.drop(columns=['ses_VMAP'], inplace=True)
  
  # Insert the column at the second position (index=1)
  df.insert(1, 'ses_VMAP', ses_VMAP_series)

  # Save the DataFrame back to the same CSV file
  df.to_csv('/nfs2/xuh11/Connectome/Demographics/VMAP_first_Epoch.csv', index=False)
