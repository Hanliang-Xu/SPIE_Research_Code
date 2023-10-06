import pandas as pd

# File paths
file1 = "/nfs2/xuh11/Connectome/CycleGAN-NumStreamlines/pytorch-CycleGAN-and-pix2pix/datasets/Num_Streamlines_1/testB/NUMSTREAMLINES_sub-019_ses-EPOCH1x210529.csv"
file2 = "/nfs2/xuh11/Connectome/CycleGan-results/VMAP/sub-019/ses-EPOCH1x210529/CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000.csv"

# Read the files into pandas DataFrames
df1 = pd.read_csv(file1, header=None)
df2 = pd.read_csv(file2, header=None)

# Make sure both dataframes have the same shape
assert df1.shape == df2.shape, "Dataframes are of different shapes"

# Subtract the dataframes
df_result = df1 - df2

# Save result to a CSV file
output_path = "/nfs2/xuh11/Connectome/difference.csv"
df_result.to_csv(output_path, header=False, index=False)
