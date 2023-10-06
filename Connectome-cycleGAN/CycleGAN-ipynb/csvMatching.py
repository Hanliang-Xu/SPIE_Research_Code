import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


VMAP_csv = "/nfs2/xuh11/Connectome/Demographics/VMAP_first_Epoch.csv"
BIOCARD_csv = "/nfs2/xuh11/Connectome/Demographics/BIOCARD_Demographics_Chenyu.csv"

VMAP_pd = pd.read_csv(VMAP_csv)
BIOCARD_pd = pd.read_csv(BIOCARD_csv)
female_count = 6
male_count = 6

# Loop for matching MCI
for i in range(VMAP_pd.shape[0]) :
    VMAP_sex = VMAP_pd.iloc[i, 2]
    VMAP_cog = VMAP_pd.iloc[i, 3]
    VMAP_age = VMAP_pd.iloc[i, 4]
    j = 1
    matched = False
    while (not matched and j < BIOCARD_pd.shape[0]) :
        BIOCARD_ID = BIOCARD_pd.iloc[j, 0]
        BIOCARD_sex = BIOCARD_pd.iloc[j, 2]
        if (VMAP_sex == BIOCARD_sex and VMAP_sex == 2):
            while (not matched and j < BIOCARD_pd.shape[0] and BIOCARD_ID == BIOCARD_pd.iloc[j, 0]):
                BIOCARD_cog = BIOCARD_pd.iloc[j, 3]
                BIOCARD_age = BIOCARD_pd.iloc[j, 4]
                if (abs(BIOCARD_age - VMAP_age) <= 0 and (VMAP_cog == "Normal")  and (BIOCARD_cog == "Normal") and (female_count < 48)):
                    if (female_count % 5 != 0) :
                        try:
                            df_concat = pd.concat([VMAP_pd.iloc[i], BIOCARD_pd.iloc[j]], axis=0).to_frame().T
                            final = pd.concat([final, df_concat], axis=0)
                            ses_BIO = pd.DataFrame({'sub': [BIOCARD_pd.iloc[j, 0]], 'ses': [BIOCARD_pd.iloc[j, 1]]})
                            final_BIO = pd.concat([final_BIO, ses_BIO], ignore_index=True)
                            ses_VMAP = pd.DataFrame({'sub': [VMAP_pd.iloc[i, 0]], 'ses': [VMAP_pd.iloc[i, 1]]})
                            final_VMAP = pd.concat([final_VMAP, ses_VMAP], ignore_index=True)
                        except:
                            final = pd.concat([VMAP_pd.iloc[i], BIOCARD_pd.iloc[j]], axis=0).to_frame().T
                            final_BIO = pd.DataFrame({'sub': [BIOCARD_pd.iloc[j, 0]], 'ses': [BIOCARD_pd.iloc[j, 1]]})
                            final_VMAP = pd.DataFrame({'sub': [VMAP_pd.iloc[i, 0]], 'ses': [VMAP_pd.iloc[i, 1]]})

                    while (j > 0 and BIOCARD_ID == BIOCARD_pd.iloc[j, 0]):
                        BIOCARD_pd.iat[j, 4] = 0
                        j -= 1
                    j += 1
                    while (j < BIOCARD_pd.shape[0] and BIOCARD_ID == BIOCARD_pd.iloc[j, 0]):
                        BIOCARD_pd.iat[j, 4] = 0
                        j += 1
                    female_count += 1
                    matched = True
                j += 1
        j += 1
    j = 0

    while (not matched and j < BIOCARD_pd.shape[0]) :
        BIOCARD_ID = BIOCARD_pd.iloc[j, 0]
        BIOCARD_sex = BIOCARD_pd.iloc[j, 2]
        if (VMAP_sex == BIOCARD_sex and VMAP_sex == 1):
            while (not matched and j < BIOCARD_pd.shape[0] and BIOCARD_ID == BIOCARD_pd.iloc[j, 0]):
                BIOCARD_cog = BIOCARD_pd.iloc[j, 3]
                BIOCARD_age = BIOCARD_pd.iloc[j, 4]
                if (abs(BIOCARD_age - VMAP_age) <= 0 and (VMAP_cog == "Normal")  and (BIOCARD_cog == "Normal")):
                    if (male_count % 5 != 0) :
                        print(male_count)
                        try:
                            df_concat = pd.concat([VMAP_pd.iloc[i], BIOCARD_pd.iloc[j]], axis=0).to_frame().T
                            final = pd.concat([final, df_concat], axis=0)
                            ses_BIO = pd.DataFrame({'sub': [BIOCARD_pd.iloc[j, 0]], 'ses': [BIOCARD_pd.iloc[j, 1]]})
                            final_BIO = pd.concat([final_BIO, ses_BIO], ignore_index=True)
                            ses_VMAP = pd.DataFrame({'sub': [VMAP_pd.iloc[i, 0]], 'ses': [VMAP_pd.iloc[i, 1]]})
                            final_VMAP = pd.concat([final_VMAP, ses_VMAP], ignore_index=True)
                        except:
                            final = pd.concat([VMAP_pd.iloc[i], BIOCARD_pd.iloc[j]], axis=0).to_frame().T
                            final_BIO = pd.DataFrame({'sub': [BIOCARD_pd.iloc[j, 0]], 'ses': [BIOCARD_pd.iloc[j, 1]]})
                            final_VMAP = pd.DataFrame({'sub': [VMAP_pd.iloc[i, 0]], 'ses': [VMAP_pd.iloc[i, 1]]})

                    while (j > 0 and BIOCARD_ID == BIOCARD_pd.iloc[j, 0]):
                        BIOCARD_pd.iat[j, 4] = 0
                        j -= 1
                    j += 1
                    while (j < BIOCARD_pd.shape[0] and BIOCARD_ID == BIOCARD_pd.iloc[j, 0]):
                        BIOCARD_pd.iat[j, 4] = 0
                        j += 1
                    matched = True
                    male_count += 1
                j += 1
        j += 1
    j = 0

    

    while (not matched and j < BIOCARD_pd.shape[0]) :
        BIOCARD_ID = BIOCARD_pd.iloc[j, 0]
        BIOCARD_sex = BIOCARD_pd.iloc[j, 2]
        if (VMAP_sex == BIOCARD_sex and VMAP_sex == 1):
            while (not matched and j < BIOCARD_pd.shape[0] and BIOCARD_ID == BIOCARD_pd.iloc[j, 0]):
                BIOCARD_cog = BIOCARD_pd.iloc[j, 3]
                BIOCARD_age = BIOCARD_pd.iloc[j, 4]
                
                if (abs(BIOCARD_age - VMAP_age) <= 1 and (VMAP_cog == "Normal")  and (BIOCARD_cog == "Normal")):
                    
                    if (male_count % 5 != 0) :
                        print(male_count)
                        try:
                            df_concat = pd.concat([VMAP_pd.iloc[i], BIOCARD_pd.iloc[j]], axis=0).to_frame().T
                            final = pd.concat([final, df_concat], axis=0)
                            ses_BIO = pd.DataFrame({'sub': [BIOCARD_pd.iloc[j, 0]], 'ses': [BIOCARD_pd.iloc[j, 1]]})
                            final_BIO = pd.concat([final_BIO, ses_BIO], ignore_index=True)
                            ses_VMAP = pd.DataFrame({'sub': [VMAP_pd.iloc[i, 0]], 'ses': [VMAP_pd.iloc[i, 1]]})
                            final_VMAP = pd.concat([final_VMAP, ses_VMAP], ignore_index=True)
                        except:
                            final = pd.concat([VMAP_pd.iloc[i], BIOCARD_pd.iloc[j]], axis=0).to_frame().T
                            final_BIO = pd.DataFrame({'sub': [BIOCARD_pd.iloc[j, 0]], 'ses': [BIOCARD_pd.iloc[j, 1]]})
                            final_VMAP = pd.DataFrame({'sub': [VMAP_pd.iloc[i, 0]], 'ses': [VMAP_pd.iloc[i, 1]]})

                    while (j > 0 and BIOCARD_ID == BIOCARD_pd.iloc[j, 0]):
                        BIOCARD_pd.iat[j, 4] = 0
                        j -= 1
                    j += 1
                    while (j < BIOCARD_pd.shape[0] and BIOCARD_ID == BIOCARD_pd.iloc[j, 0]):
                        BIOCARD_pd.iat[j, 4] = 0
                        j += 1
        
                    matched = True
                    male_count += 1
                j += 1
        j += 1
    j = 0
      

print(final)

final.to_csv("/nfs2/xuh11/Connectome/CycleGAN-ipynb/pytorch-CycleGAN-and-pix2pix/datasets/Mean_Length_5/train_set_combined.csv", index=False)
final_BIO.to_csv("/nfs2/xuh11/Connectome/CycleGAN-ipynb/pytorch-CycleGAN-and-pix2pix/datasets/Mean_Length_5/train_set_A.csv", index=False)
final_VMAP.to_csv("/nfs2/xuh11/Connectome/CycleGAN-ipynb/pytorch-CycleGAN-and-pix2pix/datasets/Mean_Length_5/train_set_B.csv", index=False)