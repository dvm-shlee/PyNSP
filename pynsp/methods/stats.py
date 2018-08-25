def linear_regression(data, estimator, design_matrix):
    import numpy as np
    if np.all(data == 0, axis=0):
        return np.zeros(data.shape)
        # return 1

    else:
        # Applying regression denoising
        model = estimator()
        model.fit(design_matrix, data)
        return model


def calc_PearsonR(signalA, signalB, decimals=3,
                  norm=True, fisher=True):
    """ Calculate Pearson R correlation coefficient between
    the time-courses extracted from the given coordinate and given signal

    :param signalA: 1D time-series data array
    :param signalB: 1D time-series data array
    :param norm: perform standard normalization
    :param fisher: perform fisher Z transdormation
    :type signalA: numpy.ndarray
    :type signalB: numpy.ndarray
    :type norm: bool
    :type fisher: bool
    :return: Pearson's correlation coefficient and it's p value
    :rtype return: r, p
    """
    from .signal import standard_norm
    from scipy.stats import pearsonr
    from numpy import arctanh
    if norm:
        signalA = standard_norm(signalA, decimals=decimals)
        signalB = standard_norm(signalB, decimals=decimals)
    r, p = pearsonr(signalA, signalB)
    if fisher:
        r = arctanh(r)
    return r, p

def map_connectivity_with_roi(img_data, roi_indices, brain_indices=None,
                              use_PCA=10, use_Bootstrap=(10, 100)):
    """

    :param img_data:
    :param roi_data:
    :param brain_indices:
    :param use_PCA:         If the value given, PCA performed.
                            the given value will be used to define number of component
    :param use_Bootstrap:   If the value given, perform Bootstrap for data sampling.
                            first value is number of voxels for sampling,
                            and second is number of iteration of Bootstrapping.
    :return:
    """
    import numpy as np

    img_array = np.asarray(img_data)
    if brain_indices is None:
        brain_indices = np.transpose(np.nonzero(img_array[:,:,:,0]))

    from .tools import extract_ts_from_coordinates
    if use_Bootstrap is not None:
        print('Using Bootstrap for re-sampling, {} voxels with {} times'.format(use_Bootstrap[0],
                                                                                use_Bootstrap[1]))
        roi_tss = extract_ts_from_coordinates(img_array, roi_indices,
                                              n_voxels=use_Bootstrap[0],
                                              iters=use_Bootstrap[1])
    else:

        roi_tss = extract_ts_from_coordinates(img_array, roi_indices)
    roi_tss = np.nan_to_num(roi_tss)

    if use_PCA is not None:
        print('Using PCA to use first principal component as signal.')
        from sklearn.decomposition import PCA
        pca = PCA(n_components=use_PCA)
        pca.fit(roi_tss)
        roi_signal = pca.components_[0, :]
    else:
        print('Using mean signal from voxels as signal.')
        roi_signal = roi_tss.mean(0)

    corr_map = np.zeros(img_array.shape[:3])
    pval_map = np.zeros(img_array.shape[:3])

    for i, t, k in brain_indices:
        R, P = calc_PearsonR(roi_signal, img_array[i, t, k], norm=True, fisher=True)
        corr_map[i, t, k] = R
        pval_map[i, t, k] = P
    return corr_map, pval_map


def map_connectivity_with_seed(img_data, coord, brain_indices=None, use_PCA=10,
                               size=1, NN=3, use_Bootstrap=(10, 100)):
    """

    :param img_data:
    :param coord:
    :param brain_indices:
    :param use_PCA:
    :param size:
    :param NN:
    :return:
    """
    import numpy as np
    img_array = np.asarray(img_data)
    if brain_indices is not None:
        brain_indices = np.transpose(np.nonzero(img_array))

    from .tools import get_cluster_coordinates, extract_ts_from_coordinates
    seed_indices = get_cluster_coordinates(coord, size=size, NN=NN, mask=brain_indices)
    if use_Bootstrap is not None:
        seed_tss = extract_ts_from_coordinates(img_array, seed_indices,
                                               n_voxels=use_Bootstrap[0],
                                               iters=use_Bootstrap[1])
    else:
        seed_tss = extract_ts_from_coordinates(img_array, seed_indices)
    if use_PCA is not None:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=use_PCA)
        pca.fit(seed_tss)
        seed_signal = pca.components_.T[0, :]
    else:
        seed_signal = seed_tss.mean(0)

    corr_map = np.empty(img_array.shape[:3])
    pval_map = np.empty(img_array.shape[:3])
    for i, t, k in brain_indices:
        R, P = calc_PearsonR(seed_signal, img_array[i, t, k], norm=True, fisher=True)
        corr_map[i, t, k] = R
        pval_map[i, t, k] = P
    return corr_map, pval_map


def multicomp_pval_correction(pvals, c_type):
    """ p value correction for Multiple-comparison
    """
    import numpy as np
    org_shape = pvals.shape
    if len(org_shape) > 1:
        pvals = pvals.flatten()
    n = pvals.shape[0]
    c_pvals = np.zeros(pvals.shape)

    if c_type == "Bonferroni":
        c_pvals = n * pvals

    elif c_type == "Bonferroni-Holm":
        values = [(pval, i) for i, pval in enumerate(pvals)]
        values.sort()
        for rank, vals in enumerate(values):
            pval, i = vals
            c_pvals[i] = (n - rank) * pval

    elif c_type == "Benjamini-Hochberg":
        values = [(pval, i) for i, pval in enumerate(pvals)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = n - i
            pval, index = vals
            new_values.append((n / rank) * pval)
        for i in xrange(0, int(n) - 1):
            if new_values[i] < new_values[i + 1]:
                new_values[i + 1] = new_values[i]
        for i, vals in enumerate(values):
            pval, index = vals
            c_pvals[index] = new_values[i]

    return c_pvals.reshape(org_shape)