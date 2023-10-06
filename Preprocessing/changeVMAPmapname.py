from pathlib import Path
import pandas as pd

VMAP_path = Path('/nfs2/xuh11/Connectome/Demographics/VMAP_first_Epoch.csv')
VMAP_df = pd.read_csv("/nfs2/xuh11/Connectome/Demographics/finalVMAP.csv")


# Define the function to apply
def pad_string(s):
    if len(s) == 1:
        return "00" + s
    elif len(s) == 2:
        return "0" + s
    else:
        return s  # Leave it unchanged if length is not 1 or 2

VMAP_df['sub'] = VMAP_df['sub'].astype(str).apply(pad_string)


#VMAP_df = VMAP_df.drop(columns='ag')
VMAP_df.to_csv("/nfs2/xuh11/Connectome/Demographics/finalVMAP.csv", index=False)