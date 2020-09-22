
"""
In this module, we can find different method to model the relationship
between stimulus and (EEG) response. Namely there are wrapper functions
implementing:

    - Forward modelling (stimulus -> EEG), a.k.a _TRF_ (Temporal Response Functions)
    - Backward modelling (EEG -> stimulus)


TODO
''''
Maybe add DNN models, if so this should rather be a subpackage.
Modules for each modelling architecture will then be implemented within the subpackage
and we would have in `__init__.py` an entry to load all architectures.
Add CCA
Add DeepCCA
Add Auction Algorithm
Add pyRiemann -> Classification?

"""

import logging
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from mne.decoding import BaseEstimator
from .utils import lag_matrix, lag_span, lag_sparse, mem_check

logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__.split('.')[0])

