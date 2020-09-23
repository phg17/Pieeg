#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:44:51 2020

@author: phg17
"""

import logging
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mne.decoding import BaseEstimator
from .utils import lag_matrix, lag_span, lag_sparse, mem_check, get_timing
from .vizu import get_spatial_colors
from scipy import linalg
import elephant
import dtw
import mne

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)
LOGGER = logging.getLogger(__name__.split('.')[0])

def _get_covmat(x,y):
    '''
    Helper function for computing auto-correlation / covariance matrices.
    '''
    return np.dot(x.T,y)


def _corr_multifeat(yhat, ytrue, nchans):
    '''
    Helper functions for computing correlation coefficient (Pearson's r) for multiple channels at once.
    Parameters
    ----------
    yhat : ndarray (T x nchan), estimate
    ytrue : ndarray (T x nchan), reference
    nchans : number of channels
    Returns
    -------
    corr_coeffs : 1-D vector (nchan), correlation coefficient for each channel
    '''
    return np.diag(np.corrcoef(x=yhat, y=ytrue, rowvar=False), k=nchans)


def _rmse_multifeat(yhat, ytrue,axis=0):
    '''
    Helper functions for computing RMSE for multiple channels at once.
    Parameters
    ----------
    yhat : ndarray (T x nchan), estimate
    ytrue : ndarray (T x nchan), reference
    axis : axis to compute the RMSE along
    Returns
    -------
    rmses : 1-D vector (nchan), RMSE for each channel
    '''
    return np.sqrt(np.mean((yhat-ytrue)**2, axis)) 


def dirac_distance(dirac1, dirac2, Fs, window_size = 0.01):
    '''
    Fast implementation of victor-purpura spike distance (faster than neo & elephant python packages)
    Direct Python port of http://www-users.med.cornell.edu/~jdvicto/pubalgor.htmlself.
    The below code was tested against the original implementation and yielded exact results.
    All credits go to the authors of the original code.
    Input:
        s1,2: pair of vectors of spike times
        cost: cost parameter for computing Victor-Purpura spike distance.
        (Note: the above need to have the same units!)
    Output:
        d: VP spike distance.
    '''
    cost = Fs * window_size
    s1 = get_timing(dirac1)
    s2 = get_timing(dirac2)

    nspi=len(s1);
    nspj=len(s2);

    scr=np.zeros((nspi+1, nspj+1));

    scr[:,0]=np.arange(nspi+1)
    scr[0,:]=np.arange(nspj+1)

    for i in np.arange(1,nspi+1):
        for j in np.arange(1,nspj+1):
            scr[i,j]=min([scr[i-1,j]+1, scr[i,j-1]+1, scr[i-1,j-1]+cost*np.abs(s1[i-1]-s2[j-1])]);

    d=scr[nspi,nspj];

    return d


def Dlp(A, B, p=2):
    cost = np.sum(np.power(np.abs(A - B), p))
    return np.power(cost, 1 / p)


def twed(A, timeSA, B, timeSB, nu, _lambda):
    # [distance, DP] = TWED( A, timeSA, B, timeSB, lambda, nu )
    # Compute Time Warp Edit Distance (TWED) for given time series A and B
    #
    # A      := Time series A (e.g. [ 10 2 30 4])
    # timeSA := Time stamp of time series A (e.g. 1:4)
    # B      := Time series B
    # timeSB := Time stamp of time series B
    # lambda := Penalty for deletion operation
    # nu     := Elasticity parameter - nu >=0 needed for distance measure
    # Reference :
    #    Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching".
    #    IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (2): 306–318. arXiv:cs/0703033
    #    http://people.irisa.fr/Pierre-Francois.Marteau/

    # Check if input arguments
    if len(A) != len(timeSA):
        print("The length of A is not equal length of timeSA")
        return None, None

    if len(B) != len(timeSB):
        print("The length of B is not equal length of timeSB")
        return None, None

    if nu < 0:
        print("nu is negative")
        return None, None

    # Add padding
    A = np.array([0] + list(A))
    timeSA = np.array([0] + list(timeSA))
    B = np.array([0] + list(B))
    timeSB = np.array([0] + list(timeSB))

    n = len(A)
    m = len(B)
    # Dynamical programming
    DP = np.zeros((n, m))

    # Initialize DP Matrix and set first row and column to infinity
    DP[0, :] = np.inf
    DP[:, 0] = np.inf
    DP[0, 0] = 0

    # Compute minimal cost
    for i in range(1, n):
        for j in range(1, m):
            # Calculate and save cost of various operations
            C = np.ones((3, 1)) * np.inf
            # Deletion in A
            C[0] = (
                DP[i - 1, j]
                + Dlp(A[i - 1], A[i])
                + nu * (timeSA[i] - timeSA[i - 1])
                + _lambda
            )
            # Deletion in B
            C[1] = (
                DP[i, j - 1]
                + Dlp(B[j - 1], B[j])
                + nu * (timeSB[j] - timeSB[j - 1])
                + _lambda
            )
            # Keep data points in both time series
            C[2] = (
                DP[i - 1, j - 1]
                + Dlp(A[i], B[j])
                + Dlp(A[i - 1], B[j - 1])
                + nu * (abs(timeSA[i] - timeSB[j]) + abs(timeSA[i - 1] - timeSB[j - 1]))
            )
            # Choose the operation with the minimal cost and update DP Matrix
            DP[i, j] = np.min(C)
    distance = DP[n - 1, m - 1]
    return distance, DP



def _ridge_fit_SVD(x, y, alpha=[0.], from_cov=False, alpha_feature = False):
    '''
    SVD-inspired fast implementation of the SVD fitting.
    Note: When fitting the intercept, it's also penalized!
          If on doesn't want that, simply use average for each channel of y to estimate intercept.
    Parameters
    ----------
    X : ndarray (nsamples x nfeats) or autocorrelation matrix XtX (nfeats x nfeats) (if from_cov == True)
    y : ndarray (nsamples x nchans) or covariance matrix XtY (nfeats x nchans) (if from_cov == True)
    alpha : array-like.
        Default: [0.].
        List of regularization parameters.
    from_cov : bool
        Default: False.
        Use covariance matrices XtX & XtY instead of raw x, y arrays.
    Returns
    -------
    model_coef : ndarray (model_feats* x alphas) *-specific shape depends on the model
    '''
    # Compute covariance matrices
    if not from_cov:
        XtX = _get_covmat(x,x)
        XtY = _get_covmat(x,y)
    else:
        XtX = x[:]
        XtY = y[:]

    # Cast alpha in ndarray
    if isinstance(alpha, float):
        alpha = np.asarray([alpha])
    elif alpha_feature:
        alpha = np.asarray(alpha).T
    else:
        alpha = np.asarray(alpha)

    # Compute eigenvalues and eigenvectors of covariance matrix XtX
    S, V = linalg.eigh(XtX, overwrite_a=False, turbo=True)

    # Sort the eigenvalues
    s_ind = np.argsort(S)[::-1]
    S = S[s_ind]
    V = V[:, s_ind]

    # Pick eigenvalues close to zero, remove them and corresponding eigenvectors
    # and compute the average
    tol = np.finfo(float).eps
    r = sum(S > tol)
    S = S[0:r]
    V = V[:, 0:r]
    nl = np.mean(S)

    # Compute z
    z = np.dot(V.T,XtY)

    # Initialize empty list to store coefficient for different regularization parameters
    coeff = []

    # Compute coefficients for different regularization parameters
    if alpha_feature:
        for l in alpha:
            coeff.append(np.dot(V, (z/(S + nl*l)[:,np.newaxis])))

    else:
        for l in alpha:
            coeff.append(np.dot(V, (z/(S[:, np.newaxis] + nl*l))))


    return np.stack(coeff, axis=-1)


class ERP_class:
    def __init__(self,tmin,tmax,srate,n_chan = 63):
        self.srate = srate
        self.tmin = tmin
        self.tmax = tmax
        self.window = lag_span(tmin,tmax,srate)
        self.times = self.window/srate
        self.ERP =np.zeros(len(self.window))
        self.mERP =np.zeros([len(self.window),n_chan])
        
    def add_data(self,eeg,events,event_type = 'spike'):

        if event_type == 'spikes':
            events_list = get_timing(events)
        else:
            events_list = events
    
        #for i in np.where(events_list < eeg.shape[0] - self.window[-1])[0]:
        for i in range(len(events_list)):
            try:
                event = events_list[i] 
                self.ERP += np.sum(np.abs(eeg[self.window + event]),axis=1)
                self.mERP += eeg[self.window + event,:]
            except:
                print('out of window')
            self.ERP = mne.filter.filter_data(self.ERP,self.srate,1,self.srate/2-1,verbose='ERROR')
            self.mERP = mne.filter.filter_data(self.mERP,self.srate,1,self.srate/2-1,verbose='ERROR')
    
    def plot_simple(self):
        plt.figure()
        plt.plot(self.times,self.ERP)
    def plot_multi(self):
        plt.figure()
        plt.plot(self.times,self.mERP)


class TRFEstimator(BaseEstimator):

    def __init__(self, times=(0.,), tmin=None, tmax=None, srate=1., alpha=[0.], fit_intercept=False, mtype='Forward'):

        # Times reflect mismatch a -> b, where a - dependent, b - predicted
        # Negative timelags indicate a lagging behind b
        # Positive timelags indicate b lagging behind a
        # For example:
        # eeg -> env (tmin = -0.5, tmax = 0.1)
        # Indicate timeframe from -100 ms (eeg precedes stimulus): 500 ms (eeg after stimulus)
        self.tmin = tmin
        self.tmax = tmax
        self.times = times
        self.srate = srate
        self.alpha = alpha
        self.mtype = mtype # Forward or backward. Required for formatting coefficients in get_coef (convention: forward - stimulus -> eeg, backward - eeg - stimulus)
        self.fit_intercept = fit_intercept
        self.fitted = False
        self.lags = None
 
        # All following attributes are only defined once fitted (hence the "_" suffix)
        self.intercept_ = None
        self.coef_ = None
        self.n_feats_ = None
        self.n_chans_ = None
        self.feat_names_ = None
        self.valid_samples_ = None
        self.XtX_ = None # Autocorrelation matrix of feature X (thus XtX) -> used for computing model using fit_from_cov 
        self.XtY_ = None # Covariance matrix of features X and Y (thus XtX) -> used for computing model using fit_from_cov 
        
        
    def fill_lags(self):
        """Fill the lags attributes, with number of samples and times in seconds.
        Note
        ----
        Necessary to call this function if one wishes to use trf.lags _before_
        :func:`trf.fit` is called.
        
        """
        if self.tmin and self.tmax:
            # LOGGER.info("Will use lags spanning form tmin to tmax.\nTo use individual lags, use the `times` argument...")
            self.lags = lag_span(self.tmin, self.tmax, srate=self.srate)[::-1] #pylint: disable=invalid-unary-operand-type
            #self.lags = lag_span(-tmax, -tmin, srate=srate) #pylint: disable=invalid-unary-operand-type
            self.times = self.lags[::-1] / self.srate
        else:
            self.times = np.asarray(self.times)
            self.lags = lag_sparse(self.times, self.srate)[::-1]
            
            
    def get_XY(self, X, y, lagged=False, drop=True, feat_names=()):
        '''
        Preprocess X and y before fitting (finding mapping between X -> y)
        Parameters
        ----------
        X : ndarray (T x nfeat)
        y : ndarray (T x nchan)
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        feat_names : list
            Names of features being fitted. Must be of length ``nfeats``.
        Returns
        -------
        Features preprocessed for fitting the model.
        X : ndarray (T x nlags * nfeats)
        y : ndarray (T x nchan)
        '''
        self.fill_lags()

        X = np.asarray(X)
        y = np.asarray(y)

        #Estimate the necessary size to compute stuff
        y_memory = sum([yy.nbytes for yy in y]) if np.ndim(y) == 3 else y.nbytes
        estimated_mem_usage = X.nbytes * (len(self.lags) if not lagged else 1) + y_memory
        if estimated_mem_usage/1024.**3 > mem_check():
            raise MemoryError("Not enough RAM available! (needed %.1fGB, but only %.1fGB available)"%(estimated_mem_usage/1024.**3, mem_check()))


        #Fill n_feat and n_chan attributes
        self.n_feats_ = X.shape[1] if not lagged else X.shape[1] // len(self.lags) #if X has been lagged, divide by number of lags
        self.n_chans_ = y.shape[1] if y.ndim == 2 else y.shape[2]

        
        #Assess if feat names corresponds to feat number
        if feat_names:
            err_msg = "Length of feature names does not match number of columns from feature matrix"
            if lagged:
                assert len(feat_names) == X.shape[1] // len(self.lags), err_msg
            else:
                assert len(feat_names) == X.shape[1], err_msg
            self.feat_names_ = feat_names

        
        n_samples_all = y.shape[0] if y.ndim == 2 else y.shape[1] # this include non-valid samples for now
        
        
        #drop samples that can't be reconstructed because on the edge, all is true otherwise
        if drop:
            self.valid_samples_ = np.logical_not(np.logical_or(np.arange(n_samples_all) < abs(max(self.lags)),
                                                               np.arange(n_samples_all)[::-1] < abs(min(self.lags))))
        else:
            self.valid_samples_ = np.ones((n_samples_all,), dtype=bool)


        # Creating lag-matrix droping NaN values if necessary
        y = y[self.valid_samples_, :] if y.ndim == 2 else y[:, self.valid_samples_, :]
        if not lagged:
            X = lag_matrix(X, lag_samples=self.lags, drop_missing=drop, filling=np.nan if drop else 0.)
            

        return X, y
    
    
    def fit(self, X, y, lagged=False, drop=True, feat_names=()):
        """Fit the TRF model.
        Mapping X -> y. Note the convention of timelags and type of model for seamless recovery of coefficients.
        Parameters
        ----------
        X : ndarray (nsamples x nfeats)
        y : ndarray (nsamples x nchans)
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        feat_names : list
            Names of features being fitted. Must be of length ``nfeats``.
        Returns
        -------
        coef_ : ndarray (alphas x nlags x nfeats)
        intercept_ : ndarray (nfeats x 1)
        """

        # Preprocess and lag inputs
        X, y = self.get_XY(X, y, lagged, drop, feat_names)

        # Adding intercept feature:
        if self.fit_intercept:
            X = np.hstack([np.ones((len(X), 1)), X])

        # Regress with Ridge to obtain coef for the input alpha
        self.coef_ = _ridge_fit_SVD(X, y, self.alpha)

        # Reshaping and getting coefficients
        if self.fit_intercept:
            self.intercept_ = self.coef_[0, np.newaxis, :]
            self.coef_ = self.coef_[1:, :]

        self.fitted = True

        return self
    
    
    def get_coef(self):
        '''
        Format and return coefficients. Note mtype attribute needs to be declared in the __init__.
        
        Returns
        -------
        coef_ : ndarray (nlags x nfeats x nchans x regularization params)
        '''
        if np.ndim(self.alpha) == 0:
            betas = np.reshape(self.coef_, (len(self.lags), self.n_feats_, self.n_chans_))
        else:
            betas = np.reshape(self.coef_, (len(self.lags), self.n_feats_, self.n_chans_, len(self.alpha)))

        if self.mtype == 'forward':
            betas = betas[::-1,:]

        return betas
    
    def add_cov(self, X, y, lagged=False, drop=True, n_parts=1):
        '''
        Compute and add (with normalization factor) covariance matrices XtX, XtY
        For v. large population models when it's not possible to load all the data to memory at once.
        Parameters
        ----------
        X : ndarray (nsamples x nfeats) or list/tuple of ndarray (from which the model will be computed)
        y : ndarray (nsamples x nchans) or list/tuple of ndarray (from which the model will be computed)
        lagged : bool
            Default: False.
            Whether the X matrix has been previously 'lagged' (intercept still to be added).
        drop : bool
            Default: True.
            Whether to drop non valid samples (if False, non valid sample are filled with 0.)
        n_parts : number of parts from which the covariance matrix are computed (required for normalization)
            Default: 1
        Returns
        -------
        XtX : autocorrelation matrix for X (accumulated)
        XtY : covariance matrix for X & Y (accumulated)
        '''
        if isinstance(X, (list,tuple)) and n_parts > 1:
            assert len(X) == len(y)
            
            for part in range(len(X)):
                assert len(X[part]) == len(y[part])
                X_part, y_part = self.get_XY(X[part], y[part], lagged, drop)
                XtX = _get_covmat(X_part, X_part)
                XtY = _get_covmat(X_part, y_part)
                norm_pool_factor = np.sqrt((n_parts*X_part.shape[0] - 1)/(n_parts*(X_part.shape[0] - 1)))

                if self.XtX_ is None:
                    self.XtX_ = XtX*norm_pool_factor
                else:
                    self.XtX_ += XtX*norm_pool_factor
                    
                if self.XtY_ is None:
                    self.XtY_ = XtY*norm_pool_factor
                else:
                    self.XtY_ += XtY*norm_pool_factor

        else:
            X_part, y_part = self.get_XY(X, y, lagged, drop)
            self.XtX = _get_covmat(X_part, X_part)
            self.XtY = _get_covmat(X_part, y_part)
        
        return self