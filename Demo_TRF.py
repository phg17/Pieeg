#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:21:39 2020

@author: phg17
"""

import numpy as np
from models import TRFEstimator
from os.path import join
import mne

path_data = 'Data'
Fs = 256
def get_raw_info(Fs = 256):
    fname = join(path_data,'info')
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.set_eeg_reference('average', projection=False)
    raw.drop_channels(['Sound','Diode','Button'])
    raw.info['sfreq'] = Fs
    return raw.info
info = get_raw_info()


envelope = np.load(join(path_data,'envelope_1.npy'))
eeg = np.load(join(path_data,'eeg_1.npy'))
vowels = np.load(join(path_data,'vowels_1.npy'))

xtrf = np.hstack([envelope,vowels])
ytrf = eeg

trf = TRFEstimator(tmin=-.5,tmax=.5,srate=Fs,alpha=[10])
trf.fit_from_cov(xtrf, ytrf,part_length=60,clear_after=False)

coef_envelope = trf.get_coef()[:,0,:,0].T
coef_vowels = trf.get_coef()[:,1,:,0].T

ev1 = mne.EvokedArray(coef_envelope,info, tmin=trf.tmin)
ev2 = mne.EvokedArray(coef_vowels,info, tmin=trf.tmin)

ev1.plot_joint()
ev2.plot_joint()

