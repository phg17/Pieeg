#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 12:21:04 2020

@author: phg17
"""

#### Libraries
import psutil
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import fftpack, signal
#from numba import jit
from sklearn.preprocessing import minmax_scale
import pandas as pd
import matplotlib.pyplot as plt
import parselmouth as pm
import mne


def RMS(signal):
    return np.sqrt(np.mean(np.power(signal, 2)))



def AddNoisePostNorm(Target, Noise, SNR, spacing=0):
    l_Target = len(Target)
    rmsS = 1
    rmsN = rmsS*(10**(-SNR/20.))
    insert = 0
    Noise = Noise[insert:insert + 2 * spacing + l_Target]
    Noise = Noise * rmsN
    Target_Noise = Noise
    Target_Noise[spacing:spacing + l_Target] += Target
    Target_Noise = Target_Noise / RMS(Target_Noise)
    
    return Target_Noise


def print_title(msg, line='=', frame=True):
    """Printing function, allowing to print a titled message (underlined or framded)

    Parameters
    ----------
    msg : str
        String of characters
    line : str
        Which character to use to underline (default "=")
    frame : bool
        Whether to frame or only underline title
    """
    print((line*len(msg)+"\n" if frame else "") + msg + '\n'+line*len(msg)+'\n')
    
    
def lag_matrix(data, lag_samples=(-1, 0, 1), filling=np.nan, drop_missing=False):
    """Helper function to create a matrix of lagged time series.

    The lag can be arbitrarily spaced. Check other functions to create series of lags
    whether they are contiguous or sparsely spanning a time window :func:`lag_span` and
    :func:`lag_sparse`.

    Parameters
    ----------
    data : ndarray (nsamples x nfeats)
        Multivariate data
    lag_samples : list
        Shift in _samples_ to be applied to data. Negative shifts are lagged in the past,
        positive shits in the future, and a shift of 0 represents the data array as it is
        in the input `data`.
    filling : float
        What value to use to fill entries which are not defined (Default: NaN).
    drop_missing : bool
        Whether to drop rows where filling occured.

    Returns
    -------
    lagged : ndarray (nsamples_new x nfeats*len(lag_samples))
        Matrix of lagged time series.

    Raises
    ------
    ValueError
        If ``filling`` is set by user and ``drop_missing`` is ``True`` (it should be one or
        the other, the error is raised to avoid this confusion by users).

    Example
    -------
    >>> data = np.asarray([[1,2,3,4,5,6],[7,8,9,10,11,12]]).T
    >>> out = lag_matrix(data, (0,1))
    >>> out
    array([[ 1.,  7.,  2.,  8.],
            [ 2.,  8.,  3.,  9.],
            [ 3.,  9.,  4., 10.],
            [ 4., 10.,  5., 11.],
            [ 5., 11.,  6., 12.],
            [ 6., 12., nan, nan]])

    """
    if not np.isnan(filling) and drop_missing:
        raise ValueError("Dropping missing values or filling them are two mutually exclusive arguments!")

    dframe = pd.DataFrame(data)

    cols = []
    for lag in lag_samples:
        #cols.append(dframe.shift(-lag))
        cols.append(dframe.shift(lag))

    dframe = pd.concat(cols, axis=1)
    dframe.fillna(filling, inplace=True)
    if drop_missing:
        dframe.dropna(inplace=True)

    return dframe.values
    #return dframe.loc[:, ::-1].get_values()
    
    

    
    
    
def lag_span(tmin, tmax, srate=125):
    """Create an array of lags spanning the time window [tmin, tmax].

    Parameters
    ----------
    tmin : float
        In seconds
    tmax : float
    srate : float
        Sampling rate

    Returns
    -------
    lags : 1d array
        Array of lags in _samples_

    """
    sample_min, sample_max = int(np.ceil(tmin * srate)), int(np.ceil(tmax * srate))
    return np.arange(sample_min, sample_max)


def lag_sparse(times, srate=125):
    """Create an array of lags for the requested time point in `times`.

    Parameters
    ----------
    times : list
        List of time point in seconds
    srate : float
        Sampling rate

    Returns
    -------
    lags : 1d array
        Array of lags in _samples_

    """
    return np.asarray([int(np.ceil(t * srate)) for t in times])


def fir_order(tbw, srate, atten=60, ripples=None):
    """Estimate FIR Type II filter order (order will be odd).

    If ripple is given will use rule:

    .. math ::

        N = \\frac{2}{3} \log_{10}\\frac{1}{10\delta_ripp\delta_att} \\frac{Fs}{TBW}

    Else:

    .. math ::

        N = \\frac{Atten*Fs}{22*TBW} - 1

    Parameters
    ----------
    tbw : float
        Transition bandwidth in Hertz
    srate : float
        Sampling rate (Fs) in Hertz
    atten : float (default 60.0)
        Attenuation in StopBand in dB
    ripples : float (default None, optional)
        Maximum ripples height (in relative to peak)

    Returns
    -------
    order : int
        Filter order (i.e. 1+numtaps)

    Notes
    -----
    Rule of thumbs from here_.

    .. _here : https://dsp.stackexchange.com/a/31077/28372
    """
    if ripples:
        atten = 10**(-abs(atten)/10.)
        order = 2./3.*np.log10(1./10/ripples/atten) * srate / tbw
    else:
        order = (atten * srate) / (22. * tbw)
        
    order = int(order)
    # be sure to return odd order
    return order + (order%2-1)




def _is_1d(arr):
    "Short utility function to check if an array is vector-like"
    return np.product(arr.shape) == max(arr.shape)


def is_pos_def(A):
    """Check if matrix is positive definite

    Ref: https://stackoverflow.com/a/44287862/5303618
    """
    if np.array_equal(A, A.conj().T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
    
    
def rolling_func(func, data, winsize=2, overlap=1, padding=True):
    """Apply a function on a rolling window on the data
    """
    #TODO: check when Parallel()(delayed(func)(x) for x in rolled_array)
    # becomes advantageous, because for now it seemed actually slower...
    # (tested only with np.cov on 15min of 64 EEG at 125Hx, list comprehension still faster)
    return [func(x) for x in chunk_data(data, win_as_samples=True, window_size=winsize, overlap_size=overlap, padding=True).swapaxes(1, 2)]

def moving_average(data, winsize=2):
    """#TODO: pad before calling chunk_data?
    """
    return chunk_data(data, window_size=winsize, overlap_size=(winsize-1)).mean(1)

def shift_array(arr, win=2, overlap=0, padding=False, axis=0):
    """Returns segments of an array (overlapping moving windows)
    using the `as_strided` function from NumPy.

    Parameters
    ----------
    arr : numpy.ndarray
    win : int
        Number of samples in one window
    overlap : int
        Number of samples overlapping (0 means no overlap)
    pad : function
        padding function to be applied to data (if False
        will throw away data)
    axis : int
        Axis on which to apply the rolling window

    Returns
    -------
    shiftarr : ndarray
        Shifted copies of array segments

    See Also
    --------
    :func:`pyeeg.utils.chunk_data`

    Notes
    -----
    Using the `as_strided` function trick from Numpy means the returned
    array share the same memory buffer as the original array, so use
    with caution!
    Maybe `.copy()` the result if needed.
    This is the way for 2d array with overlap (i.e. step size != 1, which was the easy way):

    .. code-block:: python

        as_strided(a, (num_windows, win_size, n_features), (size_onelement * hop_size * n_feats, original_strides))
    """
    n_samples = len(arr)
    if not (1 < win < n_samples):
        raise ValueError("window size must be greater than 1 and smaller than len(input)")
    if overlap < 0 or overlap > win:
        raise ValueError("Overlap size must be a positive integer smaller than window size")

    if padding:
        raise NotImplementedError("As a workaround, please pad array beforehand...")

    if not _is_1d(arr):
        if axis is not 0:
            arr = np.swapaxes(arr, 0, axis)
        return chunk_data(arr, win, overlap, padding)

    return as_strided(arr, (win, n_samples - win + 1), (arr.itemsize, arr.itemsize))


def chunk_data(data, window_size, overlap_size=0, padding=False, win_as_samples=True):
    """Nd array version of :func:`shift_array`

    Notes
    -----
    Please note that we expect first dim as our axis on which to apply
    the rolling window.
    Calling :func:`mean(axis=0)` works if ``win_as_samples`` is set to ``False``,
    otherwise use :func:`mean(axis=1)`.

    """
    assert data.ndim <= 2, "Data must be 2D at most!"
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows * window_size - (num_windows-1) * overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    # Or should I just NOT add this extra window?
    # The padding beforhand make it clear that we can handle edge values...
    if overhang != 0 and padding:
        num_windows += 1
        #newdata = np.zeros((num_windows * window_size - (num_windows-1) * overlap_size, data.shape[1]))
        #newdata[:data.shape[0]] = data
        #data = newdata
        data = np.pad(data, [(0, overhang+1), (0, 0)], mode='edge')

    size_item = data.dtype.itemsize

    if win_as_samples:
        ret = as_strided(data,
                         shape=(num_windows, window_size, data.shape[1]),
                         strides=(size_item * (window_size - overlap_size) * data.shape[1],) + data.strides)
    else:
        ret = as_strided(data,
                         shape=(window_size, num_windows, data.shape[1]),
                         strides=(data.strides[0], size_item * (window_size - overlap_size) * data.shape[1], data.strides[1]))

    return ret


def find_knee_point(x, y, tol=0.95, plot=False):
    """Function to find elbow or knee point (minimum local curvature) in a curve.
    To do so we look at the angles between adjacent segments formed by triplet of
    points.

    Parameters
    ----------
    x : 1darray
        x- coordinate of the curve
    y : 1darray
        y- coordinate of the curve
    plot : bool (default: False)
        Whether to plot the result

    Returns
    -------
    float
        The x-value of the point of maximum curvature

    Notes
    -----
    The function only works well on smooth curves.
    """
    y = np.asarray(y).copy()
    y -= y.min()
    y /= y.max()
    coords = np.asarray([x,y]).T
    local_angle = lambda v1, v2: np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    angles = []
    for k, coord in enumerate(coords[1:-1]):
        v1 = coords[k] - coord
        v2 = coords[k+2] - coord
        angles.append(local_angle(v1, v2))

    if plot:
        plt.plot(x[1:-1], minmax_scale(np.asarray(angles)/np.pi), marker='o')
        plt.hlines(tol, xmin=x[0], xmax=x[-1])
        plt.vlines(x[np.argmin(minmax_scale(np.asarray(angles)/np.pi)<=tol) + 1], ymin=0, ymax=1., linestyles='--')

    return x[np.argmin(minmax_scale(np.asarray(angles)/np.pi)<=tol) + 1]


def mem_check(units='Gb'):
    "Get available RAM"
    stats = psutil.virtual_memory()
    units = units.lower()
    if units == 'gb':
        factor = 1./1024**3
    elif units == 'mb':
        factor = 1./1024**2
    elif units == 'kb':
        factor = 1./1024
    else:
        factor = 1.
        print("Did not get what unit you want, will memory return in bytes")
    return stats.available * factor


def fast_hilbert(x, axis=0):
    '''
    Fast implementation of Hilbert transform. The trick is to find the next fast
    length of vector for fourier transform (fftpack.helper.next_fast_len(...)).
    Next the matrix of zeros of the next fast length is preallocated and filled
    with the values from the original matrix.
    Inputs:
    - x - input matrix
    - axis - axis along which the hilbert transform should be computed
    Output:
    - x - analytic signal of matrix x (the same shape, but dtype changes to np.complex)
    '''
    # Add dimension if x is a vector
    if len(x.shape) == 1:
        x = x[:, np.newaxis]

    fast_shape = np.array(
        [fftpack.helper.next_fast_len(x.shape[0]), x.shape[1]])
    x_padded = np.zeros(fast_shape)
    x_padded[:x.shape[0], :] = x
    x = signal.hilbert(x_padded, axis=axis)[:x.shape[0], :]
    return x.squeeze()



def lag_finder(y1, y2, Fs):
    n = len(y1)

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n/Fs, 0.5*n/Fs, n)
    delay = delay_arr[np.argmax(corr)]

    return int(delay*Fs)

def get_timing(spikes):
    "return timing of spikes"
    timing = []
    for i in range(len(spikes)):
        if spikes[i] == 1:
            timing.append(i)
    return timing



def compression_eeg(signal,comp_fact = 1):
    sign = np.sign(signal)
    value = np.abs(signal)**comp_fact
    return np.multiply(sign,value)


def create_events(dirac):
    timing = get_timing(dirac)
    events = np.zeros([len(timing),3])
    events[:,2] += 1
    events[:,0] = timing
    return events.astype(int)
    
def crosscorr(y1,y2,window = [-128,128],Fs = 256):
    corr = signal.correlate(y1,y2)
    corr = corr[int(len(corr)/2+window[0]):int(len(corr)/2)+window[1]]
    print(corr)
    corr = corr / corr[int(len(corr)/2)] * np.corrcoef(y1,y2)[0,1]
    print(np.corrcoef(y1,y2)[0,1])
    return corr

def filter_signal(x, srate, cutoff=None, resample=None, rescale=None, **fir_kwargs):
    """Filtering & resampling of a signal through mne create filter function.
    Args:
        x (nd array): Signal as a vector (or array - experimental).
        srate (float): Sampling rate of the signal x in Hz.
        cutoff (float | 2-element list-like, optional): Cutoff frequencies (in Hz).
            If None: no filtering. Default to None.
        resample (float, optional): Sampling rate of the resampled signal in Hz.
            If None, no resampling. Defaults to None.
        rescale ((float, float), optional): Mix-max rescale the signal to the given range.
            If None, no rescaling. Defaults to None.
        fir_kwargs (optional) - arguments of the mne.filter.create_filter
        (https://mne.tools/dev/generated/mne.filter.create_filter.html).
    Raises:
        ValueError: Incorrect formatting of input arguments.
        ValueError: Overlap of cutoff frequencies and resmapling freq.
    Returns:
        x [nd array]: Filtered (and optionally resampled) signal
    Example use:
        - Filter audio track to estimate fundamental waveforms for modelling ABR responses.
    """
    if cutoff:
        if np.isscalar(cutoff):
            l_freq = None
            h_freq = cutoff
        elif len(cutoff) == 2:
            l_freq, h_freq = cutoff
        else:
            raise ValueError(
                "Cutoffs need to be scalar (for low-pass) or 2-element vector (for bandpass).")

        f_nyq = 2*h_freq

        if not isinstance(x, float):
            x = x.astype(float)

        # Design filter
        fir_coefs = mne.filter.create_filter(
            data=x,  # data is only used for sanity checking, not strictly needed
            sfreq=srate,  # sfreq of your data in Hz
            l_freq=l_freq,
            h_freq=h_freq,  # assuming a lowpass of 40 Hz
            method='fir',
            fir_design='firwin',
            **fir_kwargs)

        # Pad & convolve
        x = np.pad(x, (len(fir_coefs) // 2, len(fir_coefs) // 2), mode='edge')
        x = signal.convolve(x, fir_coefs, mode='valid')
    else:
        f_nyq = 0

    # Resample
    if resample:
        if not f_nyq < resample <= srate:
            raise ValueError(
                "Chose resampling rate more carefully, must be > %.1f Hz" % (f_nyq))
        if srate//resample == srate/resample:
            x = signal.resample_poly(x, 1, srate//resample)
        else:
            dur = (len(x)-1)/srate
            new_n = int(np.ceil(resample * dur))
            x = signal.resample(x, new_n)

    # Scale output between 0 and 1:
    if rescale:
        x = minmax_scale(x, rescale)

    return x

def signal_envelope(x, srate, cutoff=20., resample=None, method='hilbert', comp_factor=1., rescale=None, verbose=False, **fir_kwargs):
    """Extraction of the signal envelope. + filtering and resampling.
    Args:
        x (nd array): Signal as a vector (or array - experimental).
        srate (float): Sampling rate of the signal x in Hz.
        cutoff (float | 2-element list-like, optional): Cutoff frequencies (in Hz). Defaults to 20..
        resample (float, optional): Sampling rate of the resampled signal in Hz.
            If None, no resampling. Defaults to None.
        method (str, optional): Method for extracting the envelope.
            Options:
                - hilbert - hilbert transform + abs.
                - rectify - full wave rectification.
            Defaults to 'hilbert'.
        comp_factor (float, optional): Compression factor of the envelope. Defaults to 1..
        rescale (2 element tuple of floats, optional): Mix-max rescale the signal to the given range.
            If None, no rescaling. Defaults to None.
        fir_kwargs (optional) - arguments of the mne.filter.create_filter
        (https://mne.tools/dev/generated/mne.filter.create_filter.html)
    Raises:
        NotImplementedError: Envelope extractions methods to be implemented.
        ValueError: Bad format of the argument.
    Returns:
        env [nd array]: Filtered & resampled signal envelope.
    Example use:
        - Extract envelope from speech track for modelling cortical responses.
    """

    if method.lower() == 'subs':
        raise NotImplementedError
    else:
        if method.lower() == 'hilbert':
            # Get modulus of hilbert transform
            out = abs(fast_hilbert(x))
        elif method.lower() == 'rectify':
            # Rectify x
            out = abs(x)
        else:
            raise ValueError(
                "Method can only be 'hilbert', 'rectify' or 'subs'.")

    # Non linear compression before filtering to avoid NaN
    out = out.astype(np.float)
    out = np.power(out + np.finfo(float).eps, comp_factor)

    # Filtering, resampling
    env = filter_signal(out, srate, cutoff, resample,
                        rescale, verbose=verbose, **fir_kwargs)

    return env


def signal_pitch(audio=None, srate=44100, path=None, f0_range=(50, 400), timestep=0.01, get_obj=False):
    """Estimate signal's pitch via parselmouth. Note: it's better to use original/unprocessed audio.
    Args:
        audio (1D array, optional): Speech signal. Defaults to None.
        srate (int, optional): Sampling rate of the audio file. Defaults to 44100.
        path (string, optional): Path to audio file. Defaults to None.
        f0 range ((float, float), optional): (minimal, maximal) pitch frequency (in Hz).
            Defaults to (50,400) Hz.
            Note: increasing the range slows down the method.
        timestep (float, optional): Frame size for the pitch extractor (in s). Defaults to 0.01 s.
            Note: decreasing the timestep slows down the method.
        get_obj (bool, optjonal): Return PM pitch object for further manipulation
    Returns:
        f0 (1D array): f0 frequency evolution across the recording with timestep defined above.
        pitch (pm Pitch instance): PM pitch object.
    """
    if audio is None:
        snd = pm.Sound(path)
    else:
        snd = pm.Sound(audio, srate)

    pitch = snd.to_pitch(
        pitch_floor=f0_range[0], pitch_ceiling=f0_range[1], time_step=timestep)
    f0 = pitch.selected_array['frequency'].squeeze()
    if get_obj:
        return f0, pitch
    else:
        return f0
    
def signal_f0wav(audio, srate, cutoff='auto', alpha=0.05, resample=None, **filter_kwargs):
    """Simple estimator of fundamental waveform (f0wav).
    Args:
        audio (1D array): Audio signal.
        srate (int): Sampling rate of the audio signal (in Hz).
        cutoff (str | float | array_like, optional): Cutoff frequencies of the filter.
            If 'auto' the cutoffs will be estimated from the pitch distribution as alpha
            and 1-alpha percentiles. Defaults to 'auto'.
        alpha (float, optional): Percentile of pitch used for picking cutoffs. Defaults to 0.05.
        resample (float, optional): Sampling rate of the resampled signal in Hz.
            If None, no resampling. Defaults to None.
        filter_kwargs (optional) - optional extra keyword args for filter_signal.
    Returns:
        f0wav (1D array): Estimated fundamental waveform.
    """
    f0 = signal_pitch(audio, srate)

    if isinstance(cutoff, str):
        if cutoff == 'auto':
            f0 = f0[f0 > 0]
            f0 = sorted(f0)
            lcut = f0[int(len(f0)*alpha)]
            hcut = f0[int(len(f0)*(1-alpha))]
            cutoff = [lcut, hcut]

    f0wav = filter_signal(audio, srate, cutoff, resample, **filter_kwargs)

    return f0wav

def signal_envelope_2(audio, srate, cutoff=20., method='hilbert', comp_factor=1./3, resample=1000):
    """Compute the broadband envelope of the input signal.
    Several methods are available:
        - Hilbert -> abs -> low-pass (-> resample)
        - Rectify -> low-pass (-> resample)
        - subenvelopes -> sum
    The envelope can also be compressed by raising to a certain power factor.
    Parameters
    ----------
    signal : ndarray (nsamples,)
        1-dimensional input signal
    srate : float
        Original sampling rate of the signal
    cutoff : float (default 20Hz)
        Cutoff frequency (transition will be 10 Hz)
        In Hz
    method : str {'hilbert', 'rectify', 'subs'}
        Method to be used
    comp_factor : float (default 1/3)
        Compression factor (final envelope = env**comp_factor)
    resample : float (default 125Hz)
        New sampling rate of envelope (must be 2*cutoff < .. <= srate)
        Explicitly set to False or None to skip resampling
    Returns
    -------
    env : ndarray (nsamples_env,)
        Envelope
    
    """
    print("Computing envelope...")
    if method.lower() == 'subs':
        raise NotImplementedError
    else:
        if method.lower() == 'hilbert':
            # Get modulus of hilbert transform
            out = abs(signal.hilbert(audio))
        elif method.lower() == 'rectify':
            # Rectify signal
            out = abs(audio)
        else:
            raise ValueError("Method can only be 'hilbert', 'rectify' or 'subs'.")

        # Non linear compression before filtering to avoid NaN
        out = np.power(out + np.finfo(float).eps, comp_factor)
        # Design low-pass filter
        ntaps = fir_order(10, srate, ripples=1e-3)  # + 1 -> using odd ntaps for Type I filter,
                                                    # so I have an integer group delay (instead of half)
        b = signal.firwin(ntaps, cutoff, fs=srate)
        # Filter with convolution
        out = signal.convolve(np.pad(out, (len(b) // 2, len(b) // 2), mode='edge'),
                            b, mode='valid')
        #out = scisig.filtfilt(b, [1.0], signal) # This attenuates twice as much
        #out = scisig.lfilter(b, [1.0], pad(signal, (0, len(b)//2), mode=edge))[len(b)//2:]  # slower than scipy.signal.convolve method

        # Resample
        if resample:
            if not 2*cutoff < resample <= srate:
                raise ValueError("Chose resampling rate more carefully, must be > %.1f Hz"%(cutoff))
            if srate//resample == srate/resample:
                env = signal.resample_poly(out, 1, srate//resample)
            else:
                dur = (len(audio)-1)/srate
                new_n = int(np.ceil(resample * dur))
                env = signal.resample(out, new_n)
        else:
            env = out
    
    # Scale output between 0 and 1:
    return minmax_scale(env)

def choose_regularization(accuracy, accuracy_additive, method = 'constant', optimization_objective = 'accuracy', alpha = 40, penalize = 2, plot = False):
    n_fold = accuracy.shape[0]
    
    if optimization_objective == 'accuracy':
        opti = accuracy
    elif optimization_objective == 'accuracy_additive':
        opti = accuracy_additive
    elif optimization_objective == 'gain':
        opti = accuracy - accuracy_additive
    elif optimization_objective == 'absolute_gain':
        opti = (accuracy - accuracy_additive) / np.power(accuracy + accuracy_additive + 1,penalize)
        opti_deriv = opti[:,1:] - opti[:,:-1]
    elif 'sum':
        opti = (accuracy + accuracy_additive)/2
    else:
        return 'not in function'
    
    if plot:
        plt.plot(np.mean(opti,axis=0))
        
    if method == 'constant':
        return np.repeat(alpha,n_fold)
    
    elif method == 'mean':
        if optimization_objective == 'absolute_gain':
            extrema = []
            opti_deriv = np.mean(opti_deriv,axis=0)
            for i in range(len(opti_deriv) -1):
                if np.sign(opti_deriv[i]) > np.sign(opti_deriv[i+1]):
                    extrema.append(i+1)
            if extrema:
                return np.repeat(extrema[np.argmax(np.mean(opti,axis=0)[extrema])],n_fold)
            else:
                return np.repeat(np.argmax(np.mean(opti,axis=0)),n_fold)
        else:
            return np.repeat(np.argmax(np.mean(opti,axis=0)),n_fold)
        
    elif method == 'max':
        if optimization_objective == 'absolute_gain':
            extrema = []
            opti_deriv = np.mean(opti_deriv,axis=0)
            for i in range(len(opti_deriv) -1):
                if np.sign(opti_deriv[i]) > np.sign(opti_deriv[i+1]):
                    extrema.append(i+1)
            if extrema:
                return np.repeat(extrema[np.argmax(np.mean(opti,axis=0)[extrema])],n_fold)
            else:
                return np.repeat(np.argmax(np.mean(opti,axis=0)),n_fold)
        else:
            #return np.argmax(opti,axis=1)
            return np.repeat(np.argmax(np.mean(opti,axis=0)),n_fold)
                    
    elif method == 'nested':
        values = []
        if optimization_objective == 'absolute_gain':
            for nest in range(n_fold):
                extrema = []
                folds = np.delete(np.arange(n_fold),nest)
                opti_fold = opti[folds,:]
                deriv_fold = np.mean(opti_deriv[folds,:],axis=0)
                for i in range(len(deriv_fold) -1):
                    if np.sign(deriv_fold[i]) > np.sign(deriv_fold[i+1]):
                        extrema.append(i+1)
                if extrema:
                    values.append(extrema[np.argmax(np.mean(opti_fold,axis=0)[extrema])])
                else:
                    values.append(np.argmax(np.mean(opti_fold,axis=0)))
                
        else:
            for nest in range(n_fold):
                folds = np.delete(np.arange(n_fold),nest)
                opti_fold = opti[folds,:]
                values.append(np.argmax(np.mean(opti_fold,axis=0)))
        return values
            


                        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    