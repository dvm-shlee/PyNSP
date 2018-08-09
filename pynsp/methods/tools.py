import psutil
import numpy as np


def get_cluster_coordinates(coord, size=1, NN=3, mask=None):
    """
    size: number of voxels want to include from the center
    NN: 1='faces', 2='faces and edges', 3='faces, edges, and corners'
    """
    n_voxel = size + 1
    x, y, z = coord
    X = sorted([x + i for i in range(n_voxel)] + [x - i for i in range(n_voxel) if i != 0])
    Y = sorted([y + i for i in range(n_voxel)] + [y - i for i in range(n_voxel) if i != 0])
    Z = sorted([z + i for i in range(n_voxel)] + [z - i for i in range(n_voxel) if i != 0])

    if NN is 1:
        thr = size
    elif NN is 2:
        thr = np.sqrt(np.square([size] * 2).sum())
    elif NN is 3:
        thr = np.sqrt(np.square([size] * 3).sum())
    else:
        raise Exception #TODO: Exception message handler

    all_poss = [(i, j, k) for i in X for j in Y for k in Z]
    output_coord = [c for c in all_poss if cal_distance(coord, c) <= thr]

    if mask is None:
        return output_coord
    else:
        return [c for c in output_coord if c in mask]


def cal_distance(coord1, coord2):
    return np.sqrt(np.square(np.diff(np.asarray(zip(coord1, coord2)))).sum())


def optimal_split(num, cpus, scale=10):
    result = [num%(i+1) for i in range(cpus*scale)]
    return [i+1 for i, x in enumerate(result) if x == 0][-1]


def avail_cpu():
    cpu_percent = np.sum(psutil.cpu_percent(percpu=True))
    cpu_count = psutil.cpu_count()
    return int(cpu_count * (1 - ((cpu_percent / cpu_count) * 2)/100))


def extract_ts_from_coordinates(img_data, indices, n_voxels='Max',
                                iters=None, replace=False):
    """
    Extract time-series data from ROI defined on Atlas.
    if size are provided, data will be collected from
    the randomly sampled voxels with given size.

    :param img_data: 3D+time data matrix
    :param indices: All coordinates inside the roi
    :param n_voxels: number of voxels that want to sample (default = 'Max')
    :param iters: number of iteration to perform random voxel sampling
    :param replace: Whether the sample is with or without replacement
    :type img_data:
    :type indices: 2D list
    :type n_voxels: int or 'Max'
    :type iters: int
    :type replace: boolean
    :return: 2 dimentional time-series data (averaged time-series data,
                                             number of iteration)
    :rtype return: numpy.ndarray
    """
    # The below code allows to use single coordinate index instead of set of indices
    indices = np.asarray(indices)
    if len(indices.shape) is 1:
        indices = indices[np.newaxis, :]
    else:
        indices = np.asarray(indices)
    num_ind = indices.shape[0]

    if n_voxels is 'Max':
        n_voxels = num_ind

    if iters is not None:
        result = []
        for i in range(iters):
            # result = np.zeros((img_data.shape[-1], num_ind))
            rand_index = sorted(np.random.choice(num_ind, size=n_voxels, replace=replace))
            indx, indy, indz = indices[rand_index].T.tolist()
            result.append(img_data[indx, indy, indz, :])
        result = np.concatenate(result, axis=0)
    else:
        indx, indy, indz = indices[np.random.choice(num_ind, size=n_voxels, replace=replace)].T.tolist()
        result = img_data[indx, indy, indz, :]
    return result

def save_atlas_label(label, filename):
    """ Save label instance to file

    :param label:
    :param filename:
    :return:
    """
    with open(filename, 'w') as f:
        line = list()
        for idx in label.keys():
            roi, rgb = label[idx]
            rgb = np.array(rgb) * 255
            rgb = list(rgb.astype(int))
            r, g, b = rgb
            if idx == 0:
                line = '{:>5}   {:>3}  {:>3}  {:>3}        0  0  0    "{}"\n'.format(idx, r, g, b, roi)
            else:
                line = '{}{:>5}   {:>3}  {:>3}  {:>3}        1  1  0    "{}"\n'.format(line, idx, r, g, b, roi)
        f.write(line)


def splitnifti(path):
    import os
    while '.nii' in path:
        path = os.path.splitext(path)[0]
    return str(path)


def combine_atlas(path):
    """

    :param path:
    :return:
    """
    import os
    from nibabel import Nifti1Image as ImageObj
    affine = list()

    label = dict()
    atlasdata = None
    list_of_rois = [img for img in os.listdir(path) if '.nii' in img]
    rgbs = np.random.rand(len(list_of_rois), 3)
    label[0] = 'Clear Label', [.0, .0, .0]

    for idx, img in enumerate(list_of_rois):
        imageobj = ImageObj.load(os.path.join(path, img))
        affine.append(imageobj.affine)
        if idx == 0:
            atlasdata = np.asarray(imageobj.dataobj)
        else:
            atlasdata += np.asarray(imageobj.dataobj) * (idx + 1)
        label[idx + 1] = splitnifti(img), rgbs[idx]
    atlas = ImageObj(atlasdata, affine[0])
    return atlas, label


def parsing_atlas(path):
    """Parsing atlas imageobj and label

    :param path:
    :return:
    """
    import os
    import nibabel as nib
    atlas = nib.load(path)

    filename = os.path.basename(splitnifti(path))
    dirname = os.path.dirname(path)

    label_cand = [f for f in os.listdir(dirname) if filename in f]
    if label_cand is None:
        raise Exception
    else:
        label_path = [f for f in label_cand
                      if os.path.splitext(f)[-1] in ['.lbl', '.label', '.txt']][0]
        if label_path is None:
            raise Exception
    label_dic, rgb_dic = parse_label(label_path)
    return atlas, label_dic, rgb_dic


def parse_label(path):
    label_dic = dict()
    rgb_dic = dict()
    pattern = r'^\s+(?P<idx>\d+)\s+(?P<R>\d+)\s+(?P<G>\d+)\s+(?P<B>\d+)\s+' \
              r'(\d+|\d+\.\d+)\s+\d+\s+\d+\s+"(?P<roi>.*)$'
    with open(path, 'r') as labelfile:
        import re
        for line in labelfile:
            if re.match(pattern, line):
                idx = int(re.sub(pattern, r'\g<idx>', line))
                roi = re.sub(pattern, r'\g<roi>', line)
                roi = roi.split('"')[0]
                rgb = re.sub(pattern, r'\g<R>\s\g<G>\s\g<B>', line)
                rgb = rgb.split(r'\s')
                rgb = np.array(map(float, rgb)) / 255
                label_dic[idx] = roi
                rgb_dic[idx] = rgb
    return label_dic, rgb_dic