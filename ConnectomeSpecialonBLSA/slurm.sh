#!/bin/bash
#SBATCH --mail-user=hanliang.xu@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --time=30:00:00
#SBATCH --mem=16GB
#SBATCH --array=0-1
#SBATCH --output=/nobackup/p_masi/xuh11/BLSA_ConnectomeSpecial/logs/log_%A_%a.out
#SBATCH -e /nobackup/p_masi/yangq6/BLSA_dt1/logs/logs_%A_%a.err

module load FreeSurfer/6.0.0 GCC/6.4.0-2.28 OpenMPI/2.1.1 FSL/5.0.10
module load MATLAB/2020a
export FSL_DIR=/accre/arch/easybuild/software/MPI/GCC/6.4.0-2.28/OpenMPI/2.1.1/FSL/5.0.10/fsl

mapfile -t sesses < /nobackup/p_masi/xuh11/BLSA_ConnectomeSpecial/BLSA.list
bash /nobackup/p_masi/xuh11/BLSA_ConnectomeSpecial/scripts/${sesses[$SLURM_ARRAY_TASK_ID]}.sh