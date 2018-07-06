import numpy as np
from scipy import sparse
from scipy.signal import butter, lfilter

def baseline_als(y, lamda, p, niter):
    """Asymmetric Least Squares Smoothing for Baseline fitting
    :param y: data
    :param lamda: smoothness
    :param p: assymetry
    :param niter: number of iteration
    :return:
    """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lamda * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z)

def baseline_fitting(data, lamda, p, niter=10):
    """Apply baseline fitting
    """
    z = baseline_als(data, lamda, p, niter)
    z = z - z[0]
    output = data - z
    return np.asarray(output)

def butter_bandpass(rangecut, fs, order, btype):
    nyq = 0.5 * fs
    if isinstance(rangecut, list):
        for i, cut in enumerate(rangecut):
            rangecut[i] = cut / nyq
    else:
        rangecut = rangecut / nyq
    b, a = butter(order, rangecut, btype=btype)
    return b, a

def butter_bandpass_filter(data, rangecut, fs, order=5, btype='band'):
    b, a = butter_bandpass(rangecut, fs, order=order, btype=btype)
    y = lfilter(b, a, data)
    return y

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(window_len-1)/2:-(window_len-1)/2]