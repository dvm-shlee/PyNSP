import numpy as np


def extract_ts(img_data, coord):
    """
    Extract time-series data from given image data matrix and it's coordinate

    :param img_data: 3D+time data matrix
    :param coord: 3D Euclidean coordinate
    :type img_data: numpy.ndarray
    :type coord: list
    :return: one dimentional time-series data
    :rtype return: numpy.array
    """
    return img_data[x, y, z, :]


def extract_seed_ts(matrix, tmpobj, idx, n_voxels='Max', iters=None):
    """
    Extract time-series data from ROI defined on Atlas.
    if size are provided, data will be collected from
    the randomly sampled voxels with given size.

    :param matrix: 3D+time data matrix
    :param tmpobj: TemplateObject with Atlas
    :param idx: index number of the ROI
    :param n_voxels: number of voxels that want to sample (default = 'Max')
    :param iters: number of iteration to perform random voxel sampling
    :type matrix: numpy.ndarray
    :type tmpobj: pynit.Template
    :type idx: int
    :type n_voxels: int or 'Max'
    :type iters: int
    :return: 2 dimentional time-series data (averaged time-series data,
                                             number of iteration)
    :rtype return: numpy.ndarray
    """
    seed = np.asarray(tmpobj.atlas_obj.dataobj)
    seed[seed != idx] = 0
    seed_indices = np.transpose(np.nonzero(seed))
    num_ind = seed_indices.shape[0]
    result = np.zeros((matrix.shape[-1], iters))

    if n_voxels == 'Max':
        n_voxels = num_ind
    for i in range(iters):
        idxs = seed_indices[np.random.randint(num_ind, size=n_voxels), :]
        result_ = np.zeros((matrix.shape[-1], n_voxels))
        for j in range(n_voxels):
            result_[:, j] = extract_ts(matrix, idxs[j])
        result[:, i] = result_.mean(-1)
    return result