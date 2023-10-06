"""
import pandas as pd

# Load the CSV file
df = pd.read_csv('/nfs2/xuh11/Connectome/Demographics/BIOCARD_Demographics.csv')

# Extract the fourth and fifth characters and convert them to integers
df['ses_number'] = df['ses_BIOCARD'].str[4:6].astype(int)

# Rank 'ses_number' within each 'sub_BIOCARD' group
df['rank'] = df.groupby('sub_BIOCARD')['ses_number'].rank(method='min')

# Check for duplicate numbers within each 'sub_BIOCARD' group
df['is_duplicated'] = df.duplicated(['sub_BIOCARD', 'ses_number'])

# Filter rows where 'is_duplicated' is True
duplicates = df[df['is_duplicated']]

# Print duplicate rows, if any
if not duplicates.empty:
    print("Duplicate numbers found within the same sub_BIOCARD. Ranking might not be accurate.")
else:
    print("No duplicate numbers found within the same sub_BIOCARD. Reordering rows based on rank...")
    df.sort_values(['sub_BIOCARD', 'rank'], inplace=True)
    print(df[['sub_BIOCARD', 'ses_BIOCARD', 'ses_number', 'rank']])

# Assume df is your DataFrame
df['ses_number'] = df['ses_number'] + 2000

df.to_csv('/nfs2/xuh11/Connectome/Demographics/isItCorrect.csv', index=False)

"""

import pandas as pd

# Load the CSV file
df = pd.read_csv('/nfs2/xuh11/Connectome/Demographics/BIOCARD_Demographics.csv')

# Sort by 'sub_BIOCARD' and 'age_BIOCARD'
df.sort_values(['sub_BIOCARD', 'age_BIOCARD'], inplace=True)

# Reset the index
df.reset_index(drop=True, inplace=True)

print(df)
df['year'] = df['ses_BIOCARD'].str[4:6].astype(int)
df['month'] = df['ses_BIOCARD'].str[6:8].astype(int)

df['rank'] = df.groupby('sub_BIOCARD')['year'].rank(method='min')

# Create a copy of the DataFrame's 'ses_BIOCARD' and 'rank'
df_temp = df[['sub_BIOCARD', 'ses_BIOCARD', 'rank', 'year', 'month']].copy()

# Sort by 'sub_BIOCARD' and 'rank'
df_temp.sort_values(['sub_BIOCARD', 'rank'], inplace=True)

# Reset the index
df_temp.reset_index(drop=True, inplace=True)

# Drop 'ses_BIOCARD' and 'rank' from the original DataFrame
df.drop(['ses_BIOCARD', 'rank', 'year', 'month'], axis=1, inplace=True)

# Reset the index of the original DataFrame
df.reset_index(drop=True, inplace=True)

# Concatenate the two DataFrames
df = pd.concat([df, df_temp['year']], axis=1)
df = pd.concat([df, df_temp['month']], axis=1)
df.insert(1, 'ses_BIOCARD', df_temp['ses_BIOCARD'])


#df.drop(['ses_number'], axis=1, inplace=True)
print(df)
df['age_BIOCARD'] = df['year'] + 2000 - df['birthyr'] + 0.5 + df['month'] / 12.0
df.drop(['ses_BIOCARD', 'year', 'month'], axis=1, inplace=True)
df.to_csv('/nfs2/xuh11/Connectome/Demographics/isItCorrect.csv', index=False)