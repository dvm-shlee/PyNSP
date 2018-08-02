from .io import load, save
import numpy as np


class ImageBase(object):
    """
    This is Base class for Image object
    """
    _indices_brain = None

    def __init__(self, path):
        self._img_path = path
        img = load(path)
        self._img_data = img._dataobj
        self._affine = img._affine
        self._header = img._header
        self._img_shape = img.shape
        self._img_dim = len(img.shape)

    @property
    def mask(self):
        return self._indices_brain

    @property
    def img_data(self):
        return np.asarray(self._img_data)

    @property
    def img_shape(self):
        return self._img_shape

    @property
    def img_dim(self):
        return self._img_dim

    def save_as(self, filename, key):
        save(self, filename, key=key)



class TimeSeriesBase(object):
    """
    This is Base class for Time Series object
    """

    def __init__(self, path):
        self._ts_path = path
        self._dataframe = load(path)

    @property
    def df(self):
        return self._dataframe

    def save_as(self, filename):
        save(self, filename)