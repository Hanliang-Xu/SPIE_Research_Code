import numpy as np

def print_npy_file(file_path):
    # Load the array from the .npy file
    array = np.load(file_path)

    # Print the values of the array
    print(array)

# Use the function
print_npy_file('/nfs2/xuh11/Connectome/CycleGan-results/VMAP/sub-019/ses-EPOCH1x210529/CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000.csv')
#print_npy_file('/nfs2/xuh11/Connectome/CycleGAN-ipynb/pytorch-CycleGAN-and-pix2pix/results/BIOCARDtoVMAP_Mean_Length_1/test_latest/images/MEANLENGTH_sub-JHU134547_ses-171102_fake.png.npy')