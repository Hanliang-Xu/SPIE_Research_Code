The scans are organized in the following way for VMAP:

sub-###: subject ID number

ses-EPOCH#x######:
	EPOCH#: The epoch the scan was taken in (timeline, so one epoch could correspond to 2016)
	x######: The session ID number with an 'x' separating the session ID number from the epoch number

The derivatives folder will have the same structure as the non-derivatives portion. Preprocessing that was run on the scans will be placed
in the corresponding sessions directory
	For example:

derivatives
-->sub-101
---->ses-EPOCH1x100000
------>PREQUAL
-------->...
------>WhiteMatterStamper
-------->...


Freesurfer ERRORS:
/nobackup/p_masi_brain_map/mohdkhn/VMAP/sub-334/ses-EPOCH4x239066/freesurfer_1
/nobackup/p_masi_brain_map/mohdkhn/VMAP/sub-334/ses-EPOCH3x232111/freesurfer_1
/nobackup/p_masi_brain_map/mohdkhn/VMAP/sub-056/ses-EPOCH3x233092/freesurfer_1
/nobackup/p_masi_brain_map/mohdkhn/VMAP/sub-056/ses-EPOCH4x234526/freesurfer_1
/nobackup/p_masi_brain_map/mohdkhn/VMAP/sub-251/ses-EPOCH4x238678/freesurfer_1


