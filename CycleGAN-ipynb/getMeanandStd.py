import pandas as pd

#def get_average_of_column (csv_filepath, column):

df = pd.read_csv(r'/nfs2/xuh11/Connectome/CycleGAN/datasets/train_set_combined.csv')

def printDemographicsData(df, dataset_name):
    sex_column_name = "sex_" + dataset_name
    sex = df.groupby([sex_column_name]).size()
    print("The sex distribution for", dataset_name, "is:\n", sex)

    diagnosis_column_name = "diagnosis_" + dataset_name
    diagnosis = df.groupby([diagnosis_column_name]).size()
    print("The diagnosis distribution for", dataset_name, "is:\n", diagnosis)

    age_column_name = "age_" + dataset_name
    age_mean = "{:.2f}".format(df[age_column_name].mean())
    age_median = "{:.2f}".format(df[age_column_name].median())
    age_std = "{:.2f}".format(df[age_column_name].std())
    age_max = "{:.2f}".format(df[age_column_name].max())
    age_min = "{:.2f}".format(df[age_column_name].min())
    print("The mean age for", dataset_name, "is:", age_mean)
    print("The median age for", dataset_name, "is:", age_median)
    print("The standard deviation of age for", dataset_name, "is:", age_std)
    print("The range of age for", dataset_name, "is:", end=" ")
    print(age_min, end="-")
    print(age_max)

printDemographicsData(df, "VMAP")
printDemographicsData(df, "BIOCARD")