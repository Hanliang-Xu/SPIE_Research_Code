#!/bin/bash

directory="/nfs2/harmonization/BIDS/BLSA/derivatives"
output="/nfs2/xuh11/Connectome/BLSA_diffusion"
for sub in $directory/sub-BLSA{2..9}*
do
    sub_name="${sub: -12}"
    
    for ses in "$sub"/ses*
    do
        for file in $ses/PreQualDTIdouble/PREPROCESSED/*
        do
            file_name=$(basename "$file")
            if [[ $file_name == dwmri* ]]
            then
                mkdir $output/$sub_name
                ses_name="${ses: -16}"
                mkdir $output/$sub_name/$ses_name
                cp $file $output/$sub_name/$ses_name/
            fi
        done
    done

done