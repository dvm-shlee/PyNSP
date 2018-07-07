def linear_regression(data, estimator, design_matrix):
    import numpy as np
    if np.all(data == 0, axis=0):
        return np.zeros(data.shape)

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


def option_parser_CC(SEED_options, PCA_options, Bootstrap_options):

    size, NN, mask = None, None, None
    for key, item in SEED_options:
        if key is 'size':
            size = SEED_options['size']
        if key is 'NN':
            NN = SEED_options['NN']
        if key is 'mask':
            mask = SEED_options['mask']

    n_components = None
    for key, item in PCA_options:
        if key is 'n_components':
            n_components = PCA_options['n_components']
        # other tuning options?

    n_voxels, iters, replace = None, None, None
    for key, item in Bootstrap_options:
        if key is 'n_voxels':
            n_voxels = Bootstrap_options['n_voxels']
        if key is 'iters':
            iters = Bootstrap_options['iters']
        if key is 'replace':
            replace = Bootstrap_options['replace']

    return (size, NN, mask), (n_components), (n_voxels, iters, replace)


def calc_connectivity_between_coordinates(img_data, coordA, coordB, decimals=3,
                                          use_SEED=False, SEED_options=None,
                                          use_PCA=False, PCA_options=None,
                                          use_Bootstrap=False, Bootstrap_options=None,
                                          average=True, norm=True, fisher=True):

    SEEDop, PCAop, BOOTop = option_parser_CC(SEED_options,
                                             PCA_options,
                                             Bootstrap_options)
    from .tools import extract_ts_from_coordinates
    if use_SEED is True:
        from .tools import get_cluster_coordinates
        size, NN, mask = SEEDop
        indicesA = get_cluster_coordinates(coordA, size=size, NN=NN, mask=mask)
        indicesB = get_cluster_coordinates(coordB, size=size, NN=NN, mask=mask)

        if use_Bootstrap is True:
            n_voxels, iters, replace = BOOTop
        else:
            n_voxels, iters, replace = None, None, None
        signalsA = extract_ts_from_coordinates(img_data, indicesA,
                                               n_voxels=n_voxels,
                                               iters=iters,
                                               replace=replace)
        signalsB = extract_ts_from_coordinates(img_data, indicesB,
                                               n_voxels=n_voxels,
                                               iters=iters,
                                               replace=replace)
        if use_PCA is True:
            if len(signalsA.shape) < 2:
                raise Exception #TODO: Exeption message handler
            from sklearn.decomposition import PCA
            n_components = PCA_options
            pca = PCA(n_components)
            pca.fit(signalsA)
            signalsA = pca.components_.T
            pca.fit(signalsB)
            signalsB = pca.components_.T
        if average is True:
            signalsA = signalsA.mean(1)
            signalsB = signalsB.mean(1)
        else:
            import pandas as pd
            corr_ts = pd.concat([pd.DataFrame(signalsA), pd.DataFrame(signalsB)], axis=1,
                                keys=['CoordA', 'CoordB'])
            corr_matrix = corr_ts.corr()
            if use_PCA is True:
                return corr_ts, corr_matrix
            else:
                return (corr_ts, corr_matrix), (indicesA, indicesB)

    else:
        signalsA = extract_ts_from_coordinates(img_data, coordA)
        signalsB = extract_ts_from_coordinates(img_data, coordB)

    r, p = calc_PearsonR(signalsA, signalsB, decimals=decimals,
                         norm=norm, fisher=fisher)
    return r, p






#
#
# def seedpc2brainwise(matrix, tmpobj, roi_idx, pval=None, n_pc=10,
#                      n_voxels=100, iters=1000, c_type="Benjamini-Hochberg"):
#     """brain-wise correlation analysis using Principle component of the seed
#
#     :param matrix:
#     :param tmpobj:
#     :param roi_idx:
#     :param pval:
#     :param n_pc:
#     :param n_voxels:
#     :param iters:
#     :param c_type:
#     :return:
#     """
#     seed_ts = extract_seed_ts(matrix, tmpobj, roi_idx, n_voxels, iters)
#     if n_pc:
#         pca = PCA(n_components=n_pc)
#         S_ = pca.fit_transform(seed_ts)
#     else:
#         S_ = seed_ts
#     mask = pn.load(str(tmpobj.mask))
#     indices = np.transpose(np.nonzero(mask._dataobj))
#     R, P = pearsonr_sig2img(matrix, indices, S_[:, 0], c_type)
#     if isinstance(pval, float):
#         R[P > pval] = 0
#     print('Estimation of ROI-index{} is done..'.format(str(roi_idx).zfill(3)))
#     if n_pc:
#         return np.abs(R)
#     else:
#         return R
#
#
# def pearsonr_sig2img(matrix, indices, signal, c_type):
#     """ Calculate voxel-wise pearson's correlation
#
#     :param matrix:
#     :param indices:
#     :param signal:
#     :param c_type:
#     :return:
#     """
#     x, y, z, _ = matrix.shape
#     R = np.zeros([x, y, z])
#     P = np.zeros([x, y, z])
#     for vcoord in indices:
#         xi, yi, zi = vcoord
#         R[xi, yi, zi], P[xi, yi, zi] = pearsonr_sig2vxl(matrix, vcoord, signal)
#     if c_type != None:
#         P = normalization.multicomp_pval_correction(P, c_type)
#     return R, P
#
#
# def multicomp_pval_correction(pvals, c_type):
#     """ p value correction for Multiple-comparison
#     """
#     import numpy as np
#     org_shape = pvals.shape
#     if len(org_shape) > 1:
#         pvals = pvals.flatten()
#     n = pvals.shape[0]
#     c_pvals = np.zeros(pvals.shape)
#
#     if c_type == "Bonferroni":
#         c_pvals = n * pvals
#
#     elif c_type == "Bonferroni-Holm":
#         values = [(pval, i) for i, pval in enumerate(pvals)]
#         values.sort()
#         for rank, vals in enumerate(values):
#             pval, i = vals
#             c_pvals[i] = (n - rank) * pval
#
#     elif c_type == "Benjamini-Hochberg":
#         values = [(pval, i) for i, pval in enumerate(pvals)]
#         values.sort()
#         values.reverse()
#         new_values = []
#         for i, vals in enumerate(values):
#             rank = n - i
#             pval, index = vals
#             new_values.append((n / rank) * pval)
#         for i in xrange(0, int(n) - 1):
#             if new_values[i] < new_values[i + 1]:
#                 new_values[i + 1] = new_values[i]
#         for i, vals in enumerate(values):
#             pval, index = vals
#             c_pvals[index] = new_values[i]
#
#     return c_pvals.reshape(org_shape)