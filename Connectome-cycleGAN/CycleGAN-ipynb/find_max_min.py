import os
import pandas as pd
import numpy as np

def find_min_max_in_csv(directory):
    max_value = -np.inf
    min_value = np.inf

    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                data = pd.read_csv(file_path, header=None).values

                #data = np.where(data > 0, np.log(data), data)
                
                # Set the diagonal values to zero
                #np.fill_diagonal(data, 0)
                max_value = max(max_value, data.max())
                min_value = min(min_value, data.min())

    return min_value, max_value
# Use the function
min_val, max_val = find_min_max_in_csv('/nfs2/xuh11/Connectome/CycleGAN-ipynb/pytorch-CycleGAN-and-pix2pix/datasets/Mean_Length_2')
print('Minimum Value:', min_val)
print('Maximum Value:', max_val)