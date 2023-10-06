import os
import shutil
import numpy as np

def copy_files(src_dir, dest_dir):
    # List all files in the source directory
    files = os.listdir(src_dir)

    for file in files:
        # Check if it's a file and not a directory
        if os.path.isfile(os.path.join(src_dir, file)):
            # Construct the new directory path
            # BIOCARD:
            new_dir = os.path.join(dest_dir, file[11:24], file[25:35])
            # VMAP:
            #new_dir = os.path.join(dest_dir, file[11:18], file[19:36])

            # Create the new directory, including any necessary intermediate directories
            os.makedirs(new_dir, exist_ok=True)

            # Construct the source file path
            src_file = os.path.join(src_dir, file)
        
            # Construct the destination file path
            dest_file = os.path.join(new_dir, "CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv")

            array = np.load(src_file)

            # Save the array to a .csv file without a header
            np.savetxt(dest_file, array, delimiter=",")

            # Copy the file
            # shutil.copy(src_file, dest_file)

# Call the function
copy_files('/nfs2/xuh11/Connectome/CycleGAN-ipynb/pytorch-CycleGAN-and-pix2pix/results/BIOCARDtoVMAP_Mean_Length_1/test_latest/images', '/nfs2/xuh11/Connectome/CycleGan-results/BIOCARD')