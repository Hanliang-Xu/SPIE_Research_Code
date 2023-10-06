import numpy as np

# Load the array
array = np.load('/nfs2/xuh11/Connectome/CycleGAN-ipynb/pytorch-CycleGAN-and-pix2pix/results/BIOCARDtoVMAP_MeanLength/test_latest/images/MEANLENGTH_sub-JHU020589_ses-190731_fake.png.npy')

# Print the array
print(array)
print(array.dtype)