import os
import csv

# define root directory
root_dir = "/nfs2/harmonization/raw/BLSA_ConnectomeSpecial"

# list to store the subdirectories that don't have the required file
missing_subdirectories = []

# walk through the directory tree
for dirpath, dirnames, filenames in os.walk(root_dir):
    # split the path
    path_parts = dirpath.split(os.sep)
    # check for depth 2 directories and whether the file is missing
    if len(path_parts) - len(root_dir.split(os.sep)) == 2 and "GraphMeasure_characteristicpathlength.csv" not in filenames:
        missing_subdirectories.append(path_parts[-2:])

# save as csv
with open('/nfs2/xuh11/Connectome/missingBLSA.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(['sub', 'ses'])
    # write the missing subdirectories
    for sub in missing_subdirectories:
        writer.writerow(sub)