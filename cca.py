#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:26:13 2020

@author: phg17
"""

import logging
import mne
import numpy as np
import sys
import os
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from mne.decoding import BaseEstimator
from sklearn.cross_decomposition import CCA
from pyeeg.utils import lag_matrix, lag_span, lag_sparse, is_pos_def, find_knee_point
from pyeeg.vizu import topoplot_array
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize as zscore
from pyeeg.preprocess import create_filterbank, apply_filterbank

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)