from ..tools.checker import check_dim
from ..base import np


def by_std(data):
    """ Normalize 1D time-course data
    :param data:
    :return:
    """
    if check_dim(data, dim=1):
        return (data - data.mean()) / data.std()
    else:
        return None


def multicomp_pval_correction(pvals, c_type):
    """ p value correction for Multiple-comparison
    """
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