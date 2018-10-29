def polynomial_fit(data, estimator, order=3):
    """
    Estimate polynomial curve fit for data

    :param data: time series data
    :param estimator: estimator for linear regression
    :param order: order of polynomial curve
    :return: fitted curve
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np

    model = make_pipeline(PolynomialFeatures(order), estimator())
    x = np.linspace(0, (len(data) - 1) * 2, len(data))
    X = x[:, np.newaxis]
    model.fit(X, data)
    return np.asarray(model.predict(X))


def polynomial_detrend(data, estimator, order=3):
    """
    Detrending time series data using polynomial regression

    :param data: time series data
    :param estimator: estimator for linear regression
    :param order: order of polynomial curve
    :return: detrended data
    """
    polort = polynomial_fit(data, estimator, order=order)
    return data - polort


def mode_norm(data, indices, mode=1000, decimals=None):
    """
    Mode normalization

    :param data: time series data
    :param mode: target mean value
    :param decimals: decimals for round operation
    :return: normalized data
    """
    import numpy as np
    mean_val = data.mean().copy()

    output = np.zeros(data.shape)
    for i, j, k in indices:
        ts_data = data[i, j, k, :]
        ts_data = (ts_data - mean_val) * mode / mean_val + mode
        output[i, j, k, :] = ts_data
    if decimals is not None:
        output = np.round(output, decimals)
    return output


def standard_norm(data, decimals=None, axis=0):
    """ Normalize 1D time-course data
    :param data:
    :return:
    """
    import numpy as np
    demeaned = (data - data.mean(axis))
    if demeaned.std(axis) is 0:
        return demeaned * 0
    else:
        norm_data = demeaned / demeaned.std(axis)

    if decimals is not None:
        norm_data = np.round(norm_data, decimals)
    return norm_data


def demean(data, decimals=None):
    """
    Substract mean from data

    :param data: time series data
    :param decimals: decimals for round operation
    :return: demeaned data
    """
    import numpy as np
    dm_data = data - data.mean()

    if decimals is not None:
        dm_data = np.round(dm_data, decimals)
    return dm_data


def als_fit(data, lamda, p, niter):
    """
    Asymmetric Least Squares Smoothing for Baseline or Envelope fitting

    :param data: time series data
    :param lamda: smoothness
    :param p: assymetry parameter
    :param niter: number of iteration
    :return: fitted data
    """
    import numpy as np
    from scipy import sparse

    L = len(data)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lamda * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w * data)
        w = p * (data > z) + (1 - p) * (data < z)
    return np.asarray(z)


def als_detrend(data, lamda, p, niter=10):
    """
    Apply ALS fitting

    :param data: time series data
    :param lamda: smoothness
    :param p: assymetry parameter
    :param niter: number of iteration
    :return: corrected data
    """
    import numpy as np

    z = als_fit(data, lamda, p, niter)
    z = z - z[0]
    output = data - z
    return np.asarray(output)


def freqency_filter(data, bandcut, dt, order=5, btype='band'):
    """
    Temporal frequency filter

    :param data:    time series data
    :param bandcut:   filter frequency range
    :param dt:      repeated time
    :param order:   order of the filter
    :param btype:   ({'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional)
                    The type of filter. Default is 'lowpass'.
    :return:        filtered data
    """
    import numpy as np
    from scipy.signal import butter, lfilter

    fs = 1.0/dt

    def butter_bandpass(cut_freqs, fs, order, btype):
        nyq = 0.5 * fs
        if isinstance(cut_freqs, list):
            lowcut = cut_freqs[0] / nyq
            highcut = cut_freqs[1] / nyq
            b, a = butter(order, [lowcut, highcut], btype=btype, output='ba')
        else:
            onesidecut = cut_freqs / nyq
            b, a = butter(order, onesidecut, btype=btype, output='ba')
        return b, a

    mean = data.mean()
    std = data.std()
    norm_data = standard_norm(data)

    b, a = butter_bandpass(bandcut, fs, order=order, btype=btype)
    y = lfilter(b, a, norm_data)
    return np.asarray(y) * std + mean


def window_smoothing(data, window_len=11, window='hanning'):
    """
    smooth the data using a window with requested size.
    [ref] http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html'

    :param data:        time series data
    :param window_len:  the dimension of the smoothing window; should be an odd integer
    :param window:      ({'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}, optional)
    :return:            the smoothed data
    """
    import numpy as np

    if data.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return data
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[data[window_len - 1:0:-1], data, data[-2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(window_len-1)/2:-(window_len-1)/2]


def nuisance_regression(data, estimator, ort=None, order=3):
    """

    :param data:
    :param estimator:
    :param ort:
    :param noise_pca: Use PCA as regressor if image data is provided
    :param order:
    :type noise_pca: tuple(image_data, pval)
    :return:
    """
    from .stats import linear_regression
    import pandas as pd
    import numpy as np

    polort = pd.DataFrame(polynomial_fit(data, estimator, order=order))

    if ort is None:
        design_matrix = polort
    else:
        ort = pd.DataFrame(ort)
        design_matrix = pd.concat([polort, ort, ort.diff().fillna(0)],
                                  axis=1, ignore_index=True)
    design_matrix = standard_norm(design_matrix, axis=0)

    model = linear_regression(data, estimator, design_matrix)
    if isinstance(model, np.ndarray):
        return model
    # if model == 1:
    #   return data
    else:
        regressor = model.predict(design_matrix)
        regressor -= regressor.mean()
        return np.asarray(data - regressor)


def get_pca_noise(image_data, pval):
    import numpy as np
    import scipy.stats as st
    from sklearn.decomposition import PCA

    # Norm STD map
    std_map = image_data.std(-1)
    std_map -= std_map.mean()
    std_map /= std_map.std()

    # Create the mask that shows significant noise
    std_mask = np.zeros(std_map.shape)
    std_mask[std_map >= st.norm.ppf(1 - pval)] = 1

    pca = PCA(n_components=5)
    pca.fit(image_data[np.nonzero(std_mask)])

    return pca.components_[0]