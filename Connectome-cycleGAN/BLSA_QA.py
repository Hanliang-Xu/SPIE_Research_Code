import os
import pandas as pd

# Define main directory
main_dir = "/nfs2/harmonization/raw/BLSA_ConnectomeSpecial"

# Define file names to check
files_to_check = ["CONNECTOME_Weight_MEANLENGTH_NumStreamlines_10000000.csv", "GraphMeasure_characteristicpathlength.csv"]

# Initialize dictionaries to store missing files
missing_files = {file_name: [] for file_name in files_to_check}

# Walk through directories
for root, dirs, files in os.walk(main_dir):
    # Split the root directory into parts
    parts = root.split(os.sep)
    # If depth is not 3, skip this directory
    if len(parts) - len(main_dir.split(os.sep)) != 2:
        continue

    # Otherwise, check for files
    for file_name in files_to_check:
        # Construct full file path
        file_path = os.path.join(root, file_name)
        # Check if file exists
        if not os.path.isfile(file_path):
            # If file doesn't exist, store the directory
            missing_files[file_name].append(root)

# Output directory
output_dir = "/nfs2/xuh11/Connectome"

# Write results to CSV files
for file_name, directories in missing_files.items():
    df = pd.DataFrame(directories, columns=["Missing_Directories"])
    df.to_csv(os.path.join(output_dir, f'missing_{file_name}.csv'), index=False)