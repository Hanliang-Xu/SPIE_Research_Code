import os
import csv
import itertools

dir_names = "/nfs2/harmonization/raw/BLSA_freesurfer"  # replace with your path

sub_name = []
ses_name = []
for dirpath, dirnames, filenames in os.walk(dir_names):
  # Split the path into components & get the last two directories
  parts = dirpath.split(os.sep)
  # Only consider directories at depth 1 and 2
  if len(parts) <= 7:
    full_path = os.path.join(dirpath, "freesurfer_1", "mri")
    if (len(parts) == 7) & (os.path.isdir(full_path)):
      sub_name.append(parts[-2])
      ses_name.append(parts[-1])
  else:
    # prevent os.walk from walking further down this directory tree
    dirnames[:] = []

try:
    zip_longest = itertools.zip_longest
except AttributeError:
    zip_longest = itertools.izip_longest

rows = zip_longest(sub_name, ses_name, fillvalue='')
# Writing to csv file
with open("/nfs2/xuh11/Connectome/BLSA_list_freesurfer.csv", "w") as file:
  writer = csv.writer(file)
  writer.writerows(rows)
