
# Directories
export DIFFDIR=${1} # Should be PreQualled data. Expects to be named as dwmri.nii.gz, dwmri.bvec, dwmri.bval
export FREESURFERDIR=${2}
export OUTPUTDIR=${4}
export ID=${3}

echo "NOTE: Beginning connectomics analysis with diffusion data at: ${DIFFDIR}, freesurfer output at: ${FREESURFERDIR}."
echo "NOTE: Output will be stored at ${OUTPUTDIR}"

# Hyper parameters
export NUMSTREAMS=10000000
export WORKINGDIR=/nfs2/xuh11/ConnectomeSpecialOnRAMBAM  # CHANGED by Hanliang Xu

# Set up temporary directory that will be deleted at the end of processing
mkdir ${WORKINGDIR}/temp-${ID}
export TEMPDIR=${WORKINGDIR}/temp-${ID}

# Define look up tables for atlas. This one is for desikan killany only (freesurfer default)
export LUT=/nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/FreeSurferColorLUT.txt
export FS=/nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/fs_default.txt

echo "Moving dwi to accre..."
scp xuh11@hickory.accre.vanderbilt.edu:${DIFFDIR}/* xuh11@hickory.accre.vanderbilt.edu:${TEMPDIR}

export DWI=${TEMPDIR}/dwmri.nii.gz
export BVEC=${TEMPDIR}/dwmri.bvec
export BVAL=${TEMPDIR}/dwmri.bval

echo "Getting T1 and aparc+aseg from accre (freesurfer output)..."
scp xuh11@hickory.accre.vanderbilt.edu:${FREESURFERDIR}/mri/T1.mgz xuh11@hickory.accre.vanderbilt.edu:${TEMPDIR}
scp xuh11@hickory.accre.vanderbilt.edu:${FREESURFERDIR}/mri/aparc+aseg.mgz xuh11@hickory.accre.vanderbilt.edu:${TEMPDIR}
singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif mrconvert ${TEMPDIR}/T1.mgz ${TEMPDIR}/T1.nii.gz
singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif mrconvert ${TEMPDIR}/aparc+aseg.mgz ${TEMPDIR}/aparc+aseg.nii.gz

echo "Register T1 to DWI space..."
singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif dwiextract ${DWI} -fslgrad ${BVEC} ${BVAL} - -bzero | singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif mrmath - mean ${TEMPDIR}/b0.nii.gz -axis 3
singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif antsRegistrationSyNQuick.sh -d 3 -f ${TEMPDIR}/b0.nii.gz -m ${TEMPDIR}/T1.nii.gz -o ${TEMPDIR}/T12B0 -t r
echo "Storing registered T1 at ${TEMPDIR}/T1_inDWIspace.nii.gz..."
mv ${TEMPDIR}/T12B0Warped.nii.gz ${TEMPDIR}/T1_inDWIspace.nii.gz
if test -f "${TEMPDIR}/T1_inDWIspace.nii.gz"; then
    echo "CHECK: Registered T1 found. Proceeding to next step."
else
    echo "ERROR FOUND: Registration failed. Exiting"
    exit 0;
fi


echo "Labelconvert to get the Desikan Killany atlas..."
singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif labelconvert ${TEMPDIR}/aparc+aseg.nii.gz $LUT $FS ${TEMPDIR}/atlas_freesurfer_t1.nii.gz

echo "Apply transforms to atlas image to register to subject space..."
singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif antsApplyTransforms -i ${TEMPDIR}/atlas_freesurfer_t1.nii.gz -r ${TEMPDIR}/b0.nii.gz -n NearestNeighbor -t ${TEMPDIR}/T12B00GenericAffine.mat -o ${TEMPDIR}/atlas_freesurfer_subj.nii.gz

echo "Saving atlas as ${TEMPDIR}/atlas_freesurfer_subj.nii.gz..."
export ATLAS=${TEMPDIR}/atlas_freesurfer_subj.nii.gz

echo "Estimate response functions for wm,gm, and csf..."
# Estimate response functions
#singularity exec --bind /nobackup/p_masi_brain_map/ /nobackup/p_masi_brain_map/newlinnr/diffusion.sif dwi2response dhollander ${DWI} ${TEMPDIR}/sfwm.txt ${TEMPDIR}/gm.txt ${TEMPDIR}/csf.txt -fslgrad ${BVEC} ${BVAL}
singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif dwi2response tournier ${DWI} ${TEMPDIR}/sfwm.txt -fslgrad ${BVEC} ${BVAL}

echo "Get FOD functions from the estimated response function (single fiber white matter only)..."
# Make FOD functions
echo "Checking how many shells dwi2response found..."
nr_lines=$(wc -l < ${TEMPDIR}/sfwm.txt)
if [ $nr_lines -le 4 ]; then
  echo "Single shell acquisition."
  singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif dwi2fod csd ${DWI} ${TEMPDIR}/sfwm.txt ${TEMPDIR}/wmfod.nii.gz  -fslgrad ${BVEC} ${BVAL}
else
    echo "Multishell acquisition detected."
    singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif dwi2response dhollander ${DWI} ${TEMPDIR}/sfwm.txt ${TEMPDIR}/gm.txt ${TEMPDIR}/csf.txt -fslgrad ${BVEC} ${BVAL}
    singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif dwi2fod msmt_csd ${DWI} ${TEMPDIR}/sfwm.txt ${TEMPDIR}/wmfod.nii.gz  ${TEMPDIR}/gm.txt ${TEMPDIR}/gmfod.nii.gz  ${TEMPDIR}/csf.txt ${TEMPDIR}/csffod.nii.gz -fslgrad ${BVEC} ${BVAL}
fi

echo "Use FODfs to get 5tt mask..."
# Get 5tt mask
singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif 5ttgen fsl ${TEMPDIR}/T1_inDWIspace.nii.gz ${TEMPDIR}/5tt_image.nii.gz

echo "Use 5tt mask to get the GM/WM boundary..."
# Get Grey matter -White matter boundary
singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif 5tt2gmwmi ${TEMPDIR}/5tt_image.nii.gz ${TEMPDIR}/gmwmSeed.nii.gz

echo "Start tracking using probabilistic ACT..."
# Generate 10 million streamlines
singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif tckgen -act ${TEMPDIR}/5tt_image.nii.gz -backtrack -seed_gmwmi ${TEMPDIR}/gmwmSeed.nii.gz -select ${NUMSTREAMS} ${TEMPDIR}/wmfod.nii.gz ${TEMPDIR}/tractogram_${NUMSTREAMS}.tck

echo "Save tck file as TCK_FILE=${TEMPDIR}/tractogram_${NUMSTREAMS}.tck..."
export TCK_FILE=${TEMPDIR}/tractogram_${NUMSTREAMS}.tck

echo "Map tracks to Connectomes (NOS, Mean Length), guided by atlas..."
# Map tracks to connectome (weighted by NOS)
singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif tck2connectome ${TCK_FILE} ${ATLAS} ${TEMPDIR}/CONNECTOME_Weight_NUMSTREAMLINES_NumStreamlines_${NUMSTREAMS}.csv -symmetric

# Map tracks to connectome (weighted by Mean Length of streamline)
singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/ /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/diffusion.sif  tck2connectome ${TCK_FILE} ${ATLAS} ${TEMPDIR}/CONNECTOME_Weight_MEANLENGTH_NumStreamlines_${NUMSTREAMS}.csv -scale_length -stat_edge mean -symmetric

echo "Computing graph measures"
cd /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/support_scripts
# First argument is the input directory, second argument is the output directory. Must be the absolute paths.
export COMMAND="calculategms('"${OUTPUTDIR}"','"${OUTPUTDIR}"');exit"
echo ${COMMAND}
matlab -nodisplay -nojvm -nosplash -nodesktop -r ${COMMAND}
# This script will create a csv file for each measure

singularity exec --bind /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/  /nfs2/xuh11/ConnectomeSpecialOnRAMBAM/dockers/containers_scilus_1.5.0.sif scil_compress_streamlines.py ${TCK_FILE} ${TEMPDIR}/tracks_${NUMSTREAMS}_compressed.tck
TCK_FILE=${TEMPDIR}/tracks_${NUMSTREAMS}_compressed.tck
scp -r xuh11@hickory.accre.vanderbilt.edu:${TEMPDIR}/CONNECTOME_Weight_* xuh11@hickory.accre.vanderbilt.edu:${OUTPUTDIR}/
scp -r xuh11@hickory.accre.vanderbilt.edu:${TCK_FILE} xuh11@hickory.accre.vanderbilt.edu:${OUTPUTDIR}/
scp -r xuh11@hickory.accre.vanderbilt.edu:${TEMPDIR}/GraphMeasure_* xuh11@hickory.accre.vanderbilt.edu:${OUTPUTDIR}/
echo "Removing files in ${TEMPDIR}..."

# rm -r ${TEMPDIR}