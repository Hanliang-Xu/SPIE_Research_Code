import pandas as pd

def pad_string(s):
  s = str(s)
  print(len(s))
  if len(s) == 1:
    return "sub-00" + s
  elif len(s) == 2:
    return "sub-0" + s
  else:
    return "sub-" + s  # Leave it unchanged if length is not 1 or 2
  
file_path = "/nfs2/xuh11/Connectome/Demographics/VMAP_first_Epoch.csv"
df = pd.read_csv(file_path)
df['ses'] = "ses-EPOCH1x" + df['dti.session.id']
df['map.id'] = df['map.id'].apply(pad_string)
print(df)
df.to_csv(file_path, index=False)
