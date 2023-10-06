import csv

# Define the directory to save the bash scripts
output_dir = "/nfs2/xuh11/ConnectomeSpecialOnRAMBAM/scripts/"

# Read the CSV file
with open('/nfs2/xuh11/Connectome/ConnectomeSpecialonACCRE/final_BLSA_list.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        with open(f'{output_dir}{row["sub"]}_{row["ses"]}.sh', 'w') as bash_file:
        # Create a Bash script for this row
            
            bash_file.write('#!/bin/bash\n')
            bash_file.write(f'bash /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/main.sh \\\n')
            bash_file.write(f'/nfs2/harmonization/BIDS/BLSA/derivatives/{row["sub"]}/{row["ses"]}/PreQualDTIdouble/PREPROCESSED \\\n')
            bash_file.write(f'/nfs2/harmonization/raw/BLSA_freesurfer/{row["sub"]}/{row["ses"]}/freesurfer_1 \\\n')
            bash_file.write(f'{row["sub"]}_{row["ses"]} \\\n')
            bash_file.write(f'/nfs2/harmonization/raw/BLSA_ConnectomeSpecial/{row["sub"]}/{row["ses"]}\n')
