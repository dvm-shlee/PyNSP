import numpy as np


def calc_ReHo(img_data, coord, mask=None, NN=3):
    """
    This function calculate regional homogeneity of the coordinate
    using neighboring voxels in given cluster size.

    The tied rank adjustment are performed using scipy's rankdata method

    NN: 1='faces', 2='faces and edges', 3='faces, edges, and corners'
    """

    from scipy.stats import rankdata
    from .tools import get_cluster_coordinates

    N = img_data.shape[-1]  # number of "objects"
    indices = get_cluster_coordinates(coord, size=1, NN=NN, mask=mask)
    M = len(indices)  # number of "judge"

    # Perform rank judgements
    rank_matrix = np.zeros([N, M])
    for idx, neighbor in enumerate(indices):
        i, j, k = neighbor
        try:
            rank_matrix[:, idx] = rankdata(img_data[i, j, k, :])
        except:
            # This exception handle the case that coordinate of neighbor is located outside of the matrix
            pass
    ranks = rank_matrix.sum(1)

    # Calculate the mean value of these total ranks
    mean_ranks = ranks.mean()

    # Calculate sum of squiared deviations (SSD)
    SSD = np.square(ranks - mean_ranks).sum()

    # Calculate Kendall's W
    W = 12 * SSD / (M ** 2 * (N ** 3 - N))
    return W


def map_ReHo(img_data, indices, NN=3):
    x, y, z, _ = img_data.shape
    ReHo_data = np.zeros([x, y, z])
    for i, j, k in indices:
        ReHo_data[i, j, k] = calc_ReHo(img_data, [i, j, k], NN=NN)
    return ReHo_data


def periodogram(data, dt, window='boxcar'):
    from scipy.signal import periodogram

    fs = 1.0/dt
    f, power = periodogram(data, fs, window=window)
    return f, power


def welch(data, dt, window='hann', nperseg=None, noverlap=None):
    from scipy.signal import welch
    fs = 1.0 / dt
    f, power = welch(data, fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, power


def calc_ALFF(data, dt, band=(0.01, 0.1), freq=False):
    import numpy as np
    from scipy.integrate import simps

    nperseg = 2 * (1/band[0] / dt)
    if data.shape[0] < nperseg * 2:
        f, Pspec = periodogram(data, dt)
    else:
        f, Pspec = welch(data, dt, nperseg=nperseg, noverlap=nperseg/2.0)
    if freq == True:
        return f
    else:
        low = np.argmin(abs(f - band[0]))
        high = np.argmin(abs(f - band[1]))

        ALFF = simps(Pspec[low:high], dx=1)
        ABFF = simps(Pspec, dx=1)
        if ABFF == 0:
            fALFF = 0
        else:
            fALFF = ALFF/ABFF
        return Pspec, ALFF, fALFF


def map_ALFF(img_data, indices, dt, band=(0.01, 0.1)):
    """
    map ALFF/fALFF
    :param img_data:
    :param indices:
    :param dt:
    :param band:
    :return:
    """
    from .signal import standard_norm

    x, y, z, _ = img_data.shape

    x0, y0, z0 = indices[0]
    f = calc_ALFF(img_data[x0, y0, z0, :], dt, band, freq=True)
    Pspec_data = np.zeros([x, y, z, len(f)])
    ALFF_data = np.zeros([x, y, z])
    fALFF_data = np.zeros([x, y, z])

    for idx, coord in enumerate(indices):
        i, j, k = coord
        ts_data = img_data[i, j, k, :]

        Pspec, ALFF, fALFF = calc_ALFF(ts_data, dt, band, freq=False)

        Pspec_data[i, j, k, :] = Pspec
        ALFF_data[i, j, k] = ALFF
        fALFF_data[i, j, k] = fALFF

    ALFF_data = standard_norm(ALFF_data, axis=None)
    fALFF_data = standard_norm(fALFF_data, axis=None)

    return f, Pspec_data, ALFF_data, fALFF_data