import numpy as np
import matplotlib.pyplot as plt

def getFive(file):
    # specify csv file name
    csv_filename = file

    # read the csv file (assuming it's a csv with only numerical data)
    data = np.loadtxt(csv_filename, delimiter=',')
    # Find the maximum value in the array
    max_value = np.amax(data)

    # Find the index of the maximum value in the array
    max_index = np.unravel_index(np.argmax(data), data.shape)

    print("Max value is:", max_value)
    print("Max value is at index:", max_index)

    # Set larger font sizes
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    # visualize the numpy array
    plt.figure(figsize=(10,10))  # make a 10 x 10 figure
    img = plt.imshow(data, vmin=0, vmax=200)

    # Add a colorbar
    plt.colorbar(img)
    plt.show()


getFive("/nfs2/xuh11/Connectome/CycleGan-results/VMAP/sub-019/ses-EPOCH1x210529/CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv")
getFive("/nfs2/xuh11/Connectome/CycleGan-results/VMAP/sub-034/ses-EPOCH1x209330/CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv")
getFive("/nfs2/xuh11/Connectome/CycleGan-results/VMAP/sub-036/ses-EPOCH1x209411/CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv")
getFive("/nfs2/xuh11/Connectome/CycleGan-results/VMAP/sub-050/ses-EPOCH1x210065/CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv")