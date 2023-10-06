import csv

# file paths
csv_path = '/nfs2/xuh11/Connectome/missingBLSA.csv'
txt_path = '/nfs2/xuh11/Connectome/missingBLSA.list'

with open(csv_path, 'r') as csv_file, open(txt_path, 'w') as txt_file:
    # create a csv reader
    reader = csv.reader(csv_file)
    next(reader, None)  # skip the headers
    for row in reader:
        # join the elements of the row with '_'
        txt_file.write('_'.join(row) + '\n')
