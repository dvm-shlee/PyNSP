from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from ..base import np
from ..base import pn
from ..tools.roi import extract_ts
from ..tools.roi import extract_seed_ts
from ..tools.parallel import *
import normalization


def pearsonr_sig2vxl(matrix, coord, signal,
                     norm=True, fisher=True):
    """ Calculate Pearson R correlation coefficient between
    the time-courses extracted from the given coordinate and given signal

    :param matrix: 3D+time data matrix
    :param coord: 3D Euclidean coordinate
    :param signal: 1D time-series data array
    :param norm: perform standard normalization
    :param fisher: perform fisher Z transdormation
    :type matrix: numpy.ndarray
    :type coord: list
    :type signal: numpy.ndarray
    :type norm: bool
    :type fisher: bool
    :return: 3D data matrix
    :rtype return: numpy.ndarray
    """
    ts = extract_ts(matrix, coord)
    if norm:
        ts = normalization.by_std(ts)
        signal = normalization.by_std(signal)
    r, p = pearsonr(ts, signal)
    if fisher:
        r = np.arctanh(r)
    return r, p


def pearsonr_sig2img(matrix, indices, signal, c_type):
    """ Calculate voxel-wise pearson's correlation

    :param matrix:
    :param indices:
    :param signal:
    :param c_type:
    :return:
    """
    x, y, z, _ = matrix.shape
    R = np.zeros([x, y, z])
    P = np.zeros([x, y, z])
    for vcoord in indices:
        xi, yi, zi = vcoord
        R[xi, yi, zi], P[xi, yi, zi] = pearsonr_sig2vxl(matrix, vcoord, signal)
    if c_type != None:
        P = normalization.multicomp_pval_correction(P, c_type)
    return R, P


def seedpc2brainwise(matrix, tmpobj, roi_idx, pval=None, n_pc=10,
                     n_voxels=100, iters=1000, c_type="Benjamini-Hochberg"):
    """brain-wise correlation analysis using Principle component of the seed

    :param matrix:
    :param tmpobj:
    :param roi_idx:
    :param pval:
    :param n_pc:
    :param n_voxels:
    :param iters:
    :param c_type:
    :return:
    """
    seed_ts = extract_seed_ts(matrix, tmpobj, roi_idx, n_voxels, iters)
    if n_pc:
        pca = PCA(n_components=n_pc)
        S_ = pca.fit_transform(seed_ts)
    else:
        S_ = seed_ts
    mask = pn.load(str(tmpobj.mask))
    indices = np.transpose(np.nonzero(mask._dataobj))
    R, P = pearsonr_sig2img(matrix, indices, S_[:, 0], c_type)
    if isinstance(pval, float):
        R[P > pval] = 0
    print('Estimation of ROI-index{} is done..'.format(str(roi_idx).zfill(3)))
    if n_pc:
        return np.abs(R)
    else:
        return R
