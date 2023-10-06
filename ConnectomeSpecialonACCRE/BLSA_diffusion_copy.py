import subprocess

dwi_folder = "/nfs2/harmonization/BIDS/BLSA/derivatives"
path_transfer = '/nobackup/p_masi/xuh11/BLSA_diffusion'

for sub in dwi_folder.iterdir() :
    sub_orig_folder = dwi_folder / sub
    sub_output_folder = path_transfer / sub
    subprocess.run(['mkdir','-p', sub_output_folder])
    for ses in sub_orig_folder.iterdir():
        try:
            diffusion_data = sub_orig_folder / "PreQualDTIdouble" / "PREPROCESSED"
            ses_output_folder = sub_output_folder / ses.name
            subprocess.run(['cp', diffusion_data / '*', ses_output_folder])
        except:
            continue