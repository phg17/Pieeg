#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:09:50 2021

@author: phg17
"""

import numpy as np

def letters(input):
    return ''.join(filter(str.isalpha, input))

def transfo(string):
    if string == 'JH':
        return 'TH'
    elif string == 'Y':
        return 'IY'
    

PHONES = ['AH','AE','AA','AW','AO','AY','B','CH','D','EE','EH','ER','EY','F','G','HH','IH','I','IY','DH',
          'K','L','M','N','NG','OH','OW','OY','P','R','S','SH','T','TH','UH','UW','V','W','J','Z']

Sonorant = [0,1,2,3,4,5,9,10,11,12,16,17,18,21,22,23,24,25,26,27,29,34,35,37,38]
Voiced = [0,1,2,3,4,5,6,8,9,10,11,12,14,16,17,18,19,21,22,23,24,25,26,27,29,34,35,36,37,38,39]
Syllabic = [0,1,2,3,4,5,9,10,11,12,16,17,18,25,26,27,34,35]
Obstruent = [6,7,8,13,14,15,19,20,21,22,23,24,28,29,30,31,32,33,36,39]
Labial = [6,13,22,28,36]
Coronal = [7,8,19,21,23,29,30,31,32,33,38,39]
Dorsal = [14,20,24,37]
Nasal = [22,23,24]
Stop = [6,8,14,20,28,32]
Affricate = [7,19]
Fricative = [13,15,30,31,33,36,39]
Sibilitant = [7,19,30,31,39]
Approximant = [15,21,29,37,38]
Front = [1,5,9,10,12,16,17,18]
Central = [4,5,9,11,16,26,27]
Back = [0,2,3,4,25,26,27,34,35]
Close = [5,16,17,18,34,35]
Closemid = [4,12,16,26,27]
Openmid = [1,2,3,9,10,11,12,26,27]
Open = [0,4,5,25]
Rounded = [4,5,25,26,27,34,35]

Phonetic_Features = dict()
Phonetic_Features['Sonorant'] = np.take(PHONES,Sonorant)
Phonetic_Features['Voiced'] = np.take(PHONES,Voiced)
Phonetic_Features['Syllabic'] = np.take(PHONES,Syllabic)
Phonetic_Features['Obstruent'] = np.take(PHONES,Obstruent)
Phonetic_Features['Labial'] = np.take(PHONES,Labial)
Phonetic_Features['Coronal'] = np.take(PHONES,Coronal)
Phonetic_Features['Dorsal'] = np.take(PHONES,Dorsal)
Phonetic_Features['Nasal'] = np.take(PHONES,Nasal)
Phonetic_Features['Stop'] = np.take(PHONES,Stop)
Phonetic_Features['Affricate'] = np.take(PHONES,Affricate)
Phonetic_Features['Fricative'] = np.take(PHONES,Fricative)
Phonetic_Features['Sibilitant'] = np.take(PHONES,Sibilitant)
Phonetic_Features['Approximant'] = np.take(PHONES,Approximant)
Phonetic_Features['Front'] = np.take(PHONES,Front)
Phonetic_Features['Central'] = np.take(PHONES,Central)
Phonetic_Features['Back'] = np.take(PHONES,Back)
Phonetic_Features['Close'] = np.take(PHONES,Close)
Phonetic_Features['Closemid'] = np.take(PHONES,Closemid)
Phonetic_Features['Open'] = np.take(PHONES,Open)
Phonetic_Features['Openmid'] = np.take(PHONES,Openmid)
Phonetic_Features['Rounded'] = np.take(PHONES,Rounded)