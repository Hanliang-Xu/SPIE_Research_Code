import csv

# Read the CSV file
with open('/nfs2/xuh11/Connectome/final_BLSA_list.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    
    # Create the list
    combined_list = [f'{row["sub"]}_{row["ses"]}' for row in reader]

# Write the list to a file
with open('BLSA.list', 'w') as f:
    for item in combined_list:
        f.write("%s\n" % item)

# Print the list
for item in combined_list:
    print(item)
