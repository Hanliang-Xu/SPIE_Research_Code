#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=50:00:00
#SBATCH --mem=64G
#SBATCH --output=/nobackup/p_masi/xuh11/Fran_Con_Log.out
#SBATCH --error=/nobackup/p_masi/xuh11/Fran_Con_Error.out
#SBATCH --mail-user=hanliang.xu@vanderbilt.edu
#SBATCH --mail-type=FAIL

INDIR=/nobackup/p_masi/xuh11/Fran_Con_Input/sub-2002-ses-adni3baseline/ \
OUTDIR=/nobackup/p_masi/xuh11/Fran_Con_Output/ \
singularity_path=/nobackup/p_masi/xuh11/singularity_francois_special_v1.sif \
subid=sub-2002 \
sesid=ses-adni3baseline \
runid=1 \
JOBDIR=/nobackup/p_masi/xuh11/Fran_Con_Tmp/ \
anat_path=sub-2002_ses-adni3baseline_acq-AcceleratedSagIRFSPGR_T1w.nii.gz \
dwi_path=dwmri.nii.gz \
bvec_path=dwmri.bvec \
bval_path=dwmri.bval \
seg_path=sub-2002_ses-adni3baseline_acq-AcceleratedSagIRFSPGR_T1w_seg.nii.gz \
# 0 1000 --> check bvals \
singularity run --home $JOBDIR --bind $JOBDIR:/tmp --containall --cleanenv --bind $INDIR:/INPUTS --bind $OUTDIR:/OUTPUTS --bind $JOBDIR:/TMP $singularity_path $subid $sesid $anat_path $dwi_path $bval_path $bvec_path 6 "0 1000" "0 1000" 5 wm 5 wm prob 27 0.4 20 $seg_path "4 40 41 44 45 51 52";