# Prepare PyTorch Training
# Leon Cai
# MASI Lab
# July 21, 2022

# Set Up

import torch
import torch.nn as nn
from dipy.io.streamline import load_tractogram, save_tractogram, StatefulTractogram
from dipy.io.stateful_tractogram import Space

from tqdm import tqdm
import numpy as np
import nibabel as nib
import scipy
import os
import sys

from utils import streamline2network, len2mask, triinterp

# Go!

if __name__ == '__main__':

    in_dir = sys.argv[1]
    num_streamlines = sys.argv[2]
    batch_size = sys.argv[3]

    # Parse inputs

    print('prep_pt.py: Parsing inputs...')

    assert os.path.exists(in_dir), 'Input directory does not exist.'

    trk_file = os.path.join(in_dir, 'T1_gmwmi_mni_2mm.trk')
    t1_file = os.path.join(in_dir, 'T1_N4_mni_2mm.nii.gz')

    assert os.path.exists(trk_file) and os.path.exists(t1_file), 'Input directory missing files.'

    out_dir = os.path.join(in_dir, 'dt1_gmwmi')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    num_streamlines = int(num_streamlines)
    batch_size = int(batch_size)
    num_batches = np.ceil(num_streamlines / batch_size).astype(int)

    print('prep_pt.py: Input Directory:   {}'.format(in_dir))
    print('prep_pt.py: Number of Batches: {}'.format(num_batches))
    print('prep_pt.py: Output Directory:  {}'.format(out_dir))

    # Load imaging

    print('prep_pt.py: Loading imaging...')

    t1_img = nib.load(t1_file).get_fdata()

    # Load tractography

    print('prep_pt.py: Loading tractography...')

    tractogram = load_tractogram(trk_file, reference='same', to_space=Space.VOX)
    streamlines = tractogram.streamlines
    shuffle_idxs = np.linspace(0, len(streamlines)-1, len(streamlines)).astype(int)
    np.random.shuffle(shuffle_idxs)
    streamlines = streamlines[shuffle_idxs]

    # Format streamlines

    for i, streamline_vox in tqdm(enumerate(streamlines), total=len(streamlines), desc='prep_pt.py: Formatting and saving tractography...'):
        if i % batch_size == 0:
            b = i // batch_size
            if b > 0:
                torch.save(nn.utils.rnn.pad_sequence(streamlines_step, batch_first=False), os.path.join(out_dir, 'step_{:06}.pt'.format(b-1)))
                torch.save(nn.utils.rnn.pack_sequence(streamlines_trid, enforce_sorted=False), os.path.join(out_dir, 'trid_{:06}.pt'.format(b-1)))
                torch.save(nn.utils.rnn.pack_sequence(streamlines_trii, enforce_sorted=False), os.path.join(out_dir, 'trii_{:06}.pt'.format(b-1)))
                torch.save(torch.FloatTensor(len2mask(streamlines_len)), os.path.join(out_dir, 'mask_{:06}.pt'.format(b-1)))
            streamlines_step = []
            streamlines_trid = []
            streamlines_trii = []
            streamlines_len  = []
            streamlines_tdi  = []
        cut = np.random.randint(2, streamline_vox.shape[0]-2)
        for streamline_vox_seg in [streamline_vox]: # [streamline_vox, np.flip(streamline_vox, axis=0)]: # , np.flip(streamline_vox[:cut, :], axis=0), streamline_vox[cut:, :]]:
            streamline_trid, streamline_trii, streamline_step = streamline2network(streamline_vox_seg, t1_img)
            streamlines_step.append(torch.FloatTensor(streamline_step))
            streamlines_trid.append(torch.FloatTensor(streamline_trid))
            streamlines_trii.append(torch.LongTensor(streamline_trii))
            streamlines_len.append(streamline_step.shape[0])
    torch.save(nn.utils.rnn.pad_sequence(streamlines_step, batch_first=False), os.path.join(out_dir, 'step_{:06}.pt'.format(num_batches-1)))
    torch.save(nn.utils.rnn.pack_sequence(streamlines_trid, enforce_sorted=False), os.path.join(out_dir, 'trid_{:06}.pt'.format(num_batches-1)))
    torch.save(nn.utils.rnn.pack_sequence(streamlines_trii, enforce_sorted=False), os.path.join(out_dir, 'trii_{:06}.pt'.format(num_batches-1)))
    torch.save(torch.FloatTensor(len2mask(streamlines_len)), os.path.join(out_dir, 'mask_{:06}.pt'.format(num_batches-1)))
