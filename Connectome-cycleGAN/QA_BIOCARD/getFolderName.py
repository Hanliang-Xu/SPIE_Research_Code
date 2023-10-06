import os
import csv
import itertools

dir_names = "/nfs2/xuh11/Connectome/QA_BIOCARD"  # replace with your path

sub_name = []
ses_name = []
for dirpath, dirnames, filenames in os.walk(dir_names):
  # Split the path into components & get the last two directories
  parts = dirpath.split(os.sep)
  # Only consider directories at depth 1 and 2
  if len(parts) <= 7:
    if len(parts) == 7:
      sub_name.append(parts[-2])
      ses_name.append(parts[-1])
  else:
    # prevent os.walk from walking further down this directory tree
    dirnames[:] = []

rows = itertools.zip_longest(sub_name, ses_name, fillvalue='')
# Writing to txt file
with open("/nfs2/xuh11/Connectome/QA_BIOCARD/BIOCARD_sub_ses.csv", "w", newline='') as file:
  writer = csv.writer(file)
  writer.writerows(rows)
