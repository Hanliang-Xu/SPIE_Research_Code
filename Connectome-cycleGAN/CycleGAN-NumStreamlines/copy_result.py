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
            new_dir = os.path.join(dest_dir, file[15:28], file[29:39])
            # VMAP: 
            #new_dir = os.path.join(dest_dir, file[15:22], file[23:40])

            # Create the new directory, including any necessary intermediate directories
            os.makedirs(new_dir, exist_ok=True)

            # Construct the source file path
            src_file = os.path.join(src_dir, file)

            # Construct the destination file path
            dest_file = os.path.join(new_dir, "CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_10000000.csv")

            array = np.load(src_file)

            # Save the array to a .csv file without a header
            np.savetxt(dest_file, array, delimiter=",", fmt='%i')
            # Copy the file
            # shutil.copy(src_file, dest_file)

# Call the function
copy_files('/nfs2/xuh11/Connectome/CycleGAN-NumStreamlines/pytorch-CycleGAN-and-pix2pix/results/BIOCARDtoVMAP_Num_Streamlines_1/test_latest/images', '/nfs2/xuh11/Connectome/CycleGan-results/BIOCARD')