#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 16:54:15 2020

@author: phg17
"""

import logging
import shutil
import psutil

# Installed library
import pickle
import numpy as np
import pandas as pd
import h5py # for mat file version >= 7.3
from scipy import signal as scisig
from scipy.io import loadmat, savemat
from scipy.io.wavfile import read as wavread
import os.path as ospath
#import utils
from .utils import lag_finder, AddNoisePostNorm, signal_envelope, create_events
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
# MNE:
import mne
from mne.preprocessing.ica import ICA
from mne.filter import detrend as detrend_data


File_ref = dict()
File_ref['al'] = ['al', 'al_2']
File_ref['yr'] = ['yr', 'yr_2']
File_ref['phil'] = ['phil_1', 'phil_2']
File_ref['jon'] = ['jon', 'jon_2']
File_ref['deb'] = ['deb', 'deb2']
File_ref['chap'] = ['chap_1', 'chap_2']
File_ref['alio'] = ['alio_1', 'alio_2']



Conditions_EEG = dict()
Conditions_EEG[0] = {'type':'audio','delay':0,'correlated':True} #audio only
Conditions_EEG[1] = {'type':'tactile','delay':0,'correlated':True} #tactile only
Conditions_EEG[2] = {'type':'audio-tactile','delay':60,'correlated':True} #audio tactile -60 (late tactile)
Conditions_EEG[3] = {'type':'audio-tactile','delay':0,'correlated':True} #audio tactile 0 (sync advance)
Conditions_EEG[4] = {'type':'audio-tactile','delay':-60,'correlated':True} #audio tactile 60 (tactile in advance)
Conditions_EEG[5] = {'type':'audio-tactile','delay':-120,'correlated':True} #audio tactile 120 (tactile in advance)
Conditions_EEG[6] = {'type':'audio-tactile','delay':-180,'correlated':True} #audio tactile 120 (tactile in advance)
Conditions_EEG[7] = {'type':'audio-tactile','delay':0,'correlated':False} #uncorrelated (tactile uncorrelated)

Bad_trial = dict()
Bad_trial['deb1'] = True


path_data = '/home/phg17/Documents/EEG Experiment/Data Analysis/Data'
path_stimuli = '/home/phg17/Documents/EEG Experiment/Stimuli_Odin/Stimuli' 

def load_mat(fname):
    """
    Loading simple mat file into dictionnary

    Parameters
    ----------
    fname : str
        Absolute path of file to be loaded

    Returns
    -------
    data : dict
        Dictionnary containing variables from .mat file
        (each variables is a key/value pair)
    """
    try:
        data = loadmat(fname)
    except NotImplementedError:
        print(".mat file is from matlab v7.3 or higher, will use HDF5 format.")
        with h5py.File(fname, mode='r') as fid:
            data = {}
            for k, val in fid.iteritems():
                data[k] = val.value
    return data


def get_raw_name(name, session):
    fname = ospath.join(path_data,name,'Session ' + str(session), File_ref[name][session-1] + '.vhdr')
    fpreload = ospath.join(path_data,name,'Session ' + str(session), File_ref[name][session-1] + "_preload")
    return fname, fpreload

def load_eeg_data(name, session, Fs = 1000, low_freq = 1, high_freq = 30 , ica=False, bad = True, detrend = None):
    """"
    Load eeg brainvision structure and returns data, channel names,
    sampling frequency and other useful data in a tuple

    Parameters
    ----------
    fname : str
        File path to .set EEGLAB file

    Returns
    -------
    eeg : ndarray
    srate : float
    time : ndarray
        Vector of time points
    events : ndarray
        Array of event onsets
    event_type : list
        Event type or names
    chnames : list
    stimtrack, button Press, Diode : ndarray
    """
    #if name == 'deb' and session==2:
    #    fname = ospath.join(path_data,name,'Session ' + str(session),name + str(session) + '.vhdr')
    #    fpreload = ospath.join(path_data,name,'Session ' + str(session),name + str(session) + "_preload") 
    #else:
    #    fname = ospath.join(path_data,name,'Session ' + str(session),name + '.vhdr')
    #    fpreload = ospath.join(path_data,name,'Session ' + str(session),name + "_preload")
    
    fname, fpreload = get_raw_name(name,session)
    print(fname)
    print(fpreload)
    raw = mne.io.read_raw_brainvision(fname, preload = fpreload, verbose='ERROR')
    events = mne.events_from_annotations(raw,'auto',verbose='ERROR')[0][:].T[0][1:]
    raw.set_eeg_reference('average', projection=True)
    if detrend:
        print('Detrending')
        raw_detrend = mne.io.RawArray(np.vstack([detrend_data(raw.get_data()[0:63],1,axis=1),raw.get_data()[63:]]),raw.info)
        raw = raw_detrend
    
    
    F_eeg = raw.info['sfreq']
    if Fs != F_eeg:
        raw.filter(1,Fs/2,h_trans_bandwidth=2,verbose='ERROR')
        raw.resample(Fs)
        
    raw.filter(low_freq,high_freq,h_trans_bandwidth=2,verbose='ERROR')
    
    if bad:
        raw.info['bads'] = ['Fp1', 'Fp2', 'AF8', 'AFz','CPz']
        raw.interpolate_bads()
    

    if ica:
        print('ICA is not set up properly, try to use the same shpere...etc files for everything')
        ica = ICA(n_components = 10, random_state = 97)
        ica.fit(raw)
        ica.exclude = [0]
        ica.apply(raw)
        #plt.figure()
        #ica.plot_properties()
        #ica.plot_overlay(raw, exclude=[0], picks='eeg')
    
    chnames= raw.ch_names
    time = raw.times
    srate = raw.info['sfreq']
    eeg = raw.get_data()[:63]
    #events = mne.events_from_annotations(raw,'auto',verbose='ERROR')[0][:].T[0][1:]
    stimtrack = raw['Sound'][0][0]
    button = raw['Button'][0][0]
    diode = raw['Diode'][0][0]
    #raw.drop_channels(['Sound','Diode','Button'])
    
    return chnames, time, srate, events, eeg, stimtrack, button, diode, raw.info

def load_raw_eeg_data(name, session, Fs = 1000, low_freq = 1, high_freq = 30 , ica=False):
    """"
    Load eeg brainvision structure and returns raw data

    Parameters
    ----------
    fname : str
        File path to .set EEGLAB file

    Returns
    -------
    eeg : ndarray
    srate : float
    time : ndarray
        Vector of time points
    events : ndarray
        Array of event onsets
    event_type : list
        Event type or names
    chnames : list
    stimtrack, button Press, Diode : ndarray
    """
    
    fname, fpreload = get_raw_name(name,session)
    raw = mne.io.read_raw_brainvision(fname, preload = fpreload, verbose='ERROR')
    raw.set_eeg_reference('average', projection=True)
    F_eeg = raw.info['sfreq']
    if Fs != F_eeg:
        raw.filter(1,Fs/2,h_trans_bandwidth=2,verbose='ERROR')
        raw.resample(Fs)
        
    raw.filter(low_freq,high_freq,h_trans_bandwidth=2,verbose='ERROR')
    
    print(ica)
    if ica:
        print('ICA is not set up properly, try to use the same shpere...etc files for everything')
        ica = ICA(n_components = 10, random_state = 97)
        ica.fit(raw)
        ica.exclude = [0]
        ica.apply(raw)
    
    events = mne.events_from_annotations(raw,'auto',verbose='ERROR')[0][:].T[0][1:]
    
    return raw, events



def save_raw_data(name, session, cond_list, Fs = 1000):

    
    F_resample = 38000
    cond_count = dict()
    cond_count[0] = session *2 - 2
    cond_count[1] = session *2 - 2
    cond_count[2] = session *2 - 2
    cond_count[3] = session *2 - 2
    cond_count[4] = session *2 - 2
    cond_count[5] = session *2 - 2
    cond_count[6] = session *2 - 2
    cond_count[7] = session *2 - 2
    
    fname, fpreload = get_raw_name(name,session)
    print(fname)
    print(fpreload)
    raw = mne.io.read_raw_brainvision(fname, preload = fpreload, verbose='ERROR')
    raw.set_eeg_reference('average', projection=True)
    path_save = ospath.join(path_data, 'Raw')
    
    events = mne.events_from_annotations(raw,'auto',verbose='ERROR')[0][:].T[0][1:]
    stimtrack = raw['Sound'][0][0]
    chapters, parameters = param_load(name,session)
    
    start = 0
    end = 16
    if name == 'deb' and session == 1:
        start = 1
    if name == 'deb' and session == 2:
        events = events[5:-1]
        start = 3
    if name == 'yr' and session == 1:
        end = 13
    
    for n_trial in range(start,end):
        
        chapter = chapters[n_trial//4]
        part = n_trial%4 + 1
        parameter = parameters[n_trial]
        
        if parameter in cond_list:
            audio, tactile, dirac, noise = stimuli_load(path_stimuli, chapter, part, Fs=1000)
            if name == 'deb' and session==1:
                start_trial = events[n_trial-1]  
                
            elif name == 'deb' and session==2:
                start_trial = events[n_trial-3] 
                
            else:    
                start_trial = events[n_trial]
            
            length_trial = len(audio)
            end_trial = start_trial + length_trial
            raw_current = raw.copy().crop(start_trial/1000,end_trial/1000)
            
            #stimtrack_current = stimtrack[start_trial:end_trial]
            stimtrack_current = raw_current['Sound'][0][0]
            condition = Conditions_EEG[parameter]
            tactile = np.roll(tactile,int(condition['delay']/ 1000 * Fs))
            dirac = np.roll(dirac,int(condition['delay']/ 1000 * Fs))
            audio_noise = AddNoisePostNorm(audio,noise,-2)
            print(condition)
            if condition['type'] == 'tactile':
                delay = lag_finder(stimtrack_current,tactile,Fs)
            elif condition['type'] == 'audio' or not condition['correlated']:
                delay = lag_finder(stimtrack_current,audio_noise,Fs)
            else:
                delay = lag_finder(stimtrack_current,audio_noise + tactile*2000,Fs)
            print(delay)
    
            
            audio, tactile, dirac, noise = stimuli_load(path_stimuli, chapter, part, F_resample)
            syllables = np.copy(dirac)
            dirac = np.roll(dirac,int(condition['delay']/ 1000 * F_resample))
    
            
            count = cond_count[parameter]
            cond_count[parameter] += 1
            
    
            filename = save_raw_name(name,parameter,count)
            raw_file = filename + '_raw.fif'
            stimuli_file = filename + '_stimuli.mat'
            stimuli = dict()
            stimuli['syllables'] = syllables
            stimuli['dirac'] = dirac
            stimuli['sound'] = audio
            stimuli['Fs'] = F_resample
            savemat(ospath.join(path_save,stimuli_file),stimuli)
            raw_current.save(ospath.join(path_save,raw_file))
        
def save_raw_name(name, parameter, count):
    if parameter == 0:
        filename = 'audio_' + name + '_' + str(count)
    elif parameter == 1:
        filename = 'tactile_' + name + '_' + str(count)
    elif parameter == 7:
        filename = 'sham_' + name + '_' + str(count)
    else:
        filename = 'AT_' + str(Conditions_EEG[parameter]['delay']) + '_' + name + '_' + str(count)
    return filename


def extract_duration_praat(fname):
    """Function to get the duration from a file generated by PrAat software,
    for instance to get envelope or pitch of a sound. The duration is extracted from
    the header of such files.

    Parameters
    ----------
    filepath : str
        Path of audio file

    Returns
    -------
    dur : float
        Duration of audio
    """
    with open(fname, 'r') as fid:
        headers = fid.readlines(80)
        duration = float(headers[4])
        return duration
    
    
def param_load(name, session):
    """Function to load the order of chapters and condition for each subject.

    Parameters
    ----------
    name : str
        ID of the subject
    Returns
    -------
    chapters : list
    parameters : list
    """
    
    fchapters = ospath.join(path_data,name,'Session ' + str(session),"chapters_" + str(session) + ".npy")
    fparameters = ospath.join(path_data,name,'Session ' + str(session),"parameters_" + str(session) + ".npy")
    
    chapters = np.load(fchapters)
    parameters = np.flip(np.load(fparameters))
    
    return chapters, parameters
    

def stimuli_load(path, chapter, part, Fs=39062.5):
    """Function to load the stimuli linked to a trial and resample them to the wanted frequency.

    Parameters
    ----------
    chapters : int
    part : int
    Returns
    -------
    chapters : list
    parameters : list
    """
    F_stim = 39062.5
    file = ospath.join(path,"Odin_" + str(chapter) + '_' + str(part))
    audio = np.load(file + '_audio.npy')
    tactile = np.load(file + '_phone_.npy')
    dirac = np.load(file + '_dirac.npy')
    noise = np.load(ospath.join(path, 'Odin_SSN.npy'))
    
    if Fs != F_stim:
        audio = resample(audio,39062.5,Fs)
        tactile = resample(tactile,39062.5,Fs)
        dirac = resample(dirac,39062.5,Fs,method='dirac')
        
    
    return audio, tactile, dirac, noise

    
        

def resample(data, Fs, F_resampling, method = 'continuous'):
    """Function to apply an anti-aliasing filter before resampling data

    Parameters
    ----------
    data : ndarray
    mode : decide if the signal is continuous or binary/discrete
    Returns
    -------
    out : resampled data without aliasing
    """
    
    if method == 'continuous':
        filtered_data = mne.filter.filter_data(data,Fs, l_freq = 1, h_freq = F_resampling/2., verbose = 'WARNING')
        out = mne.filter.resample(filtered_data,up = F_resampling, down = Fs, npad = 'auto')
        #ntaps = fir_order(10, Fs, ripples=1e-3)  # + 1 -> using odd ntaps for Type I filter,
                                                    # so I have an integer group delay (instead of half)
        #b = scisig.firwin(ntaps, F_resampling/2, fs=Fs)
        #filtered_data = scisig.convolve(np.pad(out, (len(b) // 2, len(b) // 2), mode='edge'),
                            #b, mode='valid')
    elif method == 'dirac':
        new_length = int(len(data) / Fs * F_resampling)+1
        out = np.zeros(new_length)
        for sample in range(len(data)):
            if data[sample] != 0:
                out[int(sample / Fs * F_resampling)] = 1
    else:
        raise ValueError("Method can only be 'continuous' or 'dirac'.")
        
    return out

        

def get_data_trial(name, session, n_trial, Fs):
    """Function to load all the information regarding a single trial of a 
    single subject. 

    Parameters
    ----------
    name : str
        ID of the subject
    n_trial : int
        number of trial, for an eeg session, there should be 16
    Fs : float
        Sampling Frequency to work with, resampling is handled already
    -------
    stimtrack : ndarray
        The stimuli as recorded by the Amplifier, useful for alignment, test
    audio : ndarray
        The audio stimuli, useful to extract features
    """
    path_stimuli = '/home/phg17/Documents/EEG Experiment/Stimuli_Odin/Stimuli' 
    
    chnames, time, srate, events, eeg, stimtrack, button, diode, info = load_eeg_data(name,session,Fs)
    chapters, parameters = param_load(name,session)
    chapter = chapters[n_trial//4]
    part = n_trial%4 + 1
    parameter = parameters[n_trial]
    
    audio, tactile, dirac, noise = stimuli_load(path_stimuli, chapter, part, Fs)
    
    start_trial = events[n_trial]
    length_trial = len(audio)
    end_trial = start_trial + length_trial
    timescale = np.arange(length_trial)
    stimtrack = mne.filter.filter_data(stimtrack[start_trial:end_trial],Fs,1,h_freq=None, verbose='ERROR')
    condition = Conditions_EEG[parameter]
    tactile = np.roll(tactile,int(condition['delay']/ 1000 * Fs))
    dirac = np.roll(dirac,int(condition['delay']/ 1000 * Fs))
    audio_noise = AddNoisePostNorm(audio,noise,-2)
    if condition['type'] == 'tactile':
        delay = lag_finder(stimtrack,tactile,Fs)
    elif condition['type'] == 'audio' or not condition['correlated']:
        delay = lag_finder(stimtrack,audio_noise,Fs)
    else:
        delay = lag_finder(stimtrack,audio_noise + tactile*2000,Fs)
  
    print(delay)
    #eeg = np.roll(eeg,-delay,axis=1)[:,start_trial:end_trial]
    #eeg = eeg[:,start_trial+delay:end_trial+delay]
    eeg = eeg[:,start_trial:end_trial]
    return stimtrack, eeg, audio, tactile, dirac, parameter, condition, timescale, info
    

def Align_and_Save(name, session, F_resample, Fs=1000, ica = False,detrend=None):
    """Function to load all the information regarding a single trial of a 
    single subject. 

    Parameters
    ----------
    name : str
        ID of the subject
    session : int
        session of recording = 1 or 2
    Fs : float
        Sampling Frequency of EEG, no resampling
    F_resample : float
        Sampling Frequency to work with and save the data as
    -------
    stimtrack : ndarray
        The stimuli as recorded by the Amplifier, useful for alignment, test
    audio : ndarray
        The audio stimuli, useful to extract features
    """
    cond_count = dict()
    cond_count[0] = 0
    cond_count[1] = 0
    cond_count[2] = 0
    cond_count[3] = 0
    cond_count[4] = 0
    cond_count[5] = 0
    cond_count[6] = 0
    cond_count[7] = 0
    
    path_save = ospath.join(path_data, str(F_resample) + 'Hz')
    chnames, time, srate, events, eeg, stimtrack, button, diode, info = load_eeg_data(name,session,Fs,ica=ica,detrend=detrend)
    chapters, parameters = param_load(name,session)
    start = 0
    end = 16
    
    for n_trial in range(start,end):
        
        chapter = chapters[n_trial//4]
        part = n_trial%4 + 1
        parameter = parameters[n_trial]
        audio, tactile, dirac, noise = stimuli_load(path_stimuli, chapter, part, Fs)
  
        start_trial = events[n_trial]
        
        length_trial = len(audio)
        end_trial = start_trial + length_trial


        stimtrack_current = stimtrack[start_trial:end_trial]
        condition = Conditions_EEG[parameter]
        tactile = np.roll(tactile,int(condition['delay']/ 1000 * Fs))
        dirac = np.roll(dirac,int(condition['delay']/ 1000 * Fs))
        audio_noise = AddNoisePostNorm(audio,noise,-2)
        
        print(condition)
        if condition['type'] == 'tactile':
            delay = lag_finder(stimtrack_current,tactile,Fs)
        elif condition['type'] == 'audio' or not condition['correlated']:
            delay = lag_finder(stimtrack_current,audio_noise,Fs)
        else:
            delay = lag_finder(stimtrack_current,audio_noise + tactile*2000,Fs)
        print(delay)
        #eeg = np.roll(eeg,-delay,axis=1)[:,start_trial:end_trial]
        #eeg = eeg[:,start_trial+delay:end_trial+delay]
        #eeg_current = eeg[:,start_trial+delay:end_trial+delay]
        eeg_current = eeg[:,start_trial:end_trial]
        
        
        trial = dict()
        audio, tactile, dirac, noise = stimuli_load(path_stimuli, chapter, part, F_resample)
        syllables = np.copy(dirac)
        tactile = np.roll(tactile,int(condition['delay']/ 1000 * F_resample))
        dirac = np.roll(dirac,int(condition['delay']/ 1000 * F_resample))
        envelope = signal_envelope(audio, F_resample, cutoff=20, method='hilbert', resample = None)
        eeg_current = mne.filter.resample(eeg_current,up = F_resample, down = Fs)
        trial['condition'] = condition
        trial['response'] = eeg_current
        trial['audio'] = audio
        trial['envelope'] = envelope
        trial['dirac'] = dirac[:len(envelope)]
        trial['syllables'] = syllables[:len(envelope)]
        trial['tactile'] = tactile[:len(envelope)]
        
        
        count = cond_count[parameter]
        cond_count[parameter] += 1
        
        filename = get_name(parameter,name,session,count,F_resample,ica)
        file = ospath.join(path_save,filename)
        print(file)
    
        output = open(file, 'wb')
        pickle.dump(trial, output)
        output.close()
        
        
def get_name(parameter,name,session,count,Fs,ica = False, erp = False):
    path_save = ospath.join(path_data, str(Fs) + 'Hz')
    condition = Conditions_EEG[parameter]
    if not ica:
        if not condition['correlated']:
            filename = 'uncorrelated_' + name + '_' + str(session) + '_' + str(Fs) + 'Hz_' + str(count) + '_' +  '.pkl'  
        elif condition['type'] == 'audio-tactile':
            filename = 'audio_tactile_' + str(condition['delay']) + '_' + name + '_' + str(session) + '_' + str(Fs) + 'Hz_' + str(count) + '_' +  '.pkl'  
        else:
            filename = str(condition['type']) + '_' + name + '_' + str(session) + '_' + str(Fs) + 'Hz_' + str(count) + '_' +  '.pkl'  
    else:
        if not condition['correlated']:
            filename = 'uncorrelated_ica_' + name + '_' + str(session) + '_' + str(Fs) + 'Hz_' + str(count) + '_' +  '.pkl'  
        elif condition['type'] == 'audio-tactile':
            filename = 'audio_tactile_ica_' + str(condition['delay']) + '_' + name + '_' + str(session) + '_' + str(Fs) + 'Hz_' + str(count) + '_' +  '.pkl'  
        else:
            filename = str(condition['type']) + '_ica_' + name + '_' + str(session) + '_' + str(Fs) + 'Hz_' + str(count) + '_' +  '.pkl'  
    
    if erp:
        filename = 'erp_' + filename
    file = ospath.join(path_save,filename)
    
    return file

def get_raw_info():
    fname = ospath.join(path_data,'info')
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.set_eeg_reference('average', projection=True)
    raw.drop_channels(['Sound','Diode','Button'])
    return raw.info


def Generate_Arrays(name_list,sessions,parameter_list,Fs,non_lin=1,ica=False,erp=False,concatenate=True):
    
    envelopes = []
    diracs = []
    syllables_list = []
    tactiles = []
    eegs = []
    envelopes2 = []
    for name in name_list:
        for session in sessions:
            for i in range(2):
                for parameter in parameter_list:  
                    try:
                        file = get_name(parameter,name,session,i,Fs,ica=ica,erp=False)
                        pkl_file = open(file, 'rb')
                        trial = pickle.load(pkl_file)
                        pkl_file.close()
                        
                        envelope = trial['envelope']
                        envelope2 = np.copy(envelope)
                        envelope2 = np.reshape(scale(envelope2.T),(len(envelope2),1))
                        envelopes2.append(envelope2)
                        
                        envelope -= np.min(envelope)
                        envelope /= np.max(envelope)
                        envelope = np.power(envelope,non_lin)
                        envelope = np.reshape(scale(envelope.T),(len(envelope),1))
                        envelope -= np.min(envelope)
                        #envelope = np.power(envelope,non_lin)
                        envelopes.append(envelope)
    
                        
                        dirac = trial['dirac']
                        dirac = np.reshape(scale(dirac.T),(len(dirac),1))
                        dirac /= np.max(dirac)
                        diracs.append(dirac[:len(envelope)])
                        
                        syllables = trial['syllables']
                        syllables = np.reshape(scale(syllables.T),(len(syllables),1))
                        syllables /= np.max(syllables)
                        syllables_list.append(syllables[:len(envelope)])
                        
                        tactile = trial['tactile']
                        tactile = np.reshape(scale(tactile.T),(len(tactile),1))
                        tactiles.append(tactile[:len(envelope)])
                        
                        eeg = trial['response'].T                    
                        eegs.append(eeg)
                        
                    except:
                        print('missing trial for condition number' + str(parameter) + ' for ' + name)
    total_eeg =[]
    total_eeg.append(eegs)
    if concatenate:
        y1 = np.concatenate(np.concatenate(total_eeg,axis=0),axis=0)
        #y2 = utils.compression_eeg(y1,comp_fact=1/3)
        #y3 = mne.filter.filter_data(y1,Fs,1,4,verbose='ERROR')
        #y4 = mne.filter.filter_data(y1,Fs,4,8,verbose='ERROR')
        #y5 = mne.filter.filter_data(y1,Fs,8,16,verbose='ERROR')
        #y6 = mne.filter.filter_data(y1,Fs,1,16,verbose='ERROR')
        #y7 = mne.filter.filter_data(y1,Fs,l_freq=None,h_freq=50,verbose='ERROR')
        y = y1


        x1 = np.concatenate(envelopes)[:len(y)]
        #x2 = (np.concatenate(diracs)[:len(y)]/np.max(diracs[0])).astype(int)
        #x3 = (np.concatenate(syllables_list)[:len(y)]/np.max(syllables_list[0])).astype(int)
        x2 = (np.concatenate(diracs)[:len(y)]).astype(int)
        x3 = (np.concatenate(syllables_list)[:len(y)]).astype(int)
        x4 = np.concatenate(tactiles)[:len(y)]
        x5 = np.concatenate(envelopes2)[:len(y)]
    else:
        y = total_eeg[0]
        x1 = envelopes
        x2 = diracs
        x3 = syllables_list
        x4 = tactiles
        x5 = envelopes2
    
    return y,x1,x2,x3,x4,x5
    











def Tactile_ERP(name_list, session, F_resample, Fs=1000, t_min = -1., t_max = 1., ref_epoch = 'tactile', ica = False, detrend = None):


    
    epochs_dict = dict()
    evoked_dict = dict()
    for i in range(8):
        epochs_dict[i] = []
        evoked_dict[i] = []
    
    name = name_list[0]
    
    for name in name_list:
        events_dict = dict()
        for i in range(8):
            events_dict[i] = np.asarray([[],[],[]]).T
            
        raw, events = load_raw_eeg_data(name,session,F_resample,ica=ica)
        chapters, parameters = param_load(name,session)
        start = 0
        end = 16
        if name == 'deb':
            start = 1
        if name == 'yr':
            end = 13
        
        for n_trial in range(start,end):
            
            chapter = chapters[n_trial//4]
            part = n_trial%4 + 1
            parameter = parameters[n_trial]
            audio, tactile, dirac, noise = stimuli_load(path_stimuli, chapter, part, F_resample)
            if name == 'deb':
                start_trial = events[n_trial-1]
                
            else:    
                start_trial = events[n_trial]
            
            #length_trial = len(audio)
            #end_trial = start_trial + length_trial
            condition = Conditions_EEG[parameter]
            #raw_current = raw.crop(start_trial/F_resample,end_trial/F_resample)
            if ref_epoch == 'tactile':
                dirac = np.roll(dirac,int(condition['delay']/ 1000 * F_resample))

            
            eve = create_events(dirac) 
            eve[:,0] += start_trial + raw.first_samp
            events_dict[parameter] = np.vstack([events_dict[parameter],eve]).astype(int)
        
        for i in range(8):
            print(name,i)
            epochs_dict[i].append(mne.Epochs(raw,events_dict[i],tmin=t_min,tmax=t_max,event_id={'pulse':1}, detrend=detrend,preload = False,reject = None, verbose='ERROR'))
    for i in range(8):
        print(name,i)
        epochs_dict[i] = mne.concatenate_epochs(epochs_dict[i])
        epochs_dict[i].drop_channels(['Sound','Diode','Button'])
        evoked_dict[i].append(epochs_dict[i].average())
    return epochs_dict, evoked_dict

'''            
            epochs = mne.Epochs(raw, eve, tmin=-1., tmax=1.,event_id={'pulse':1} ,preload=False, reject=None)
            evoked = epochs['pulse'].average()
            print(condition)
    return evoked, epochs

'''            
            
    

    
    