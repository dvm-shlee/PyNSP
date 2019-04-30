from .io import load, save
import numpy as np
import os

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


class MatlabEngBase(object):
    """
    This is Base class for Matlab Engine
    """

    def __init__(self, *args):
        # Initiate engine
        try:
            import matlab.engine
        except:
            raise Exception #TODO: Error message to show no matlab engine

        from pynsp import package_directory
        self.path = os.path.join(package_directory, *args)
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(self.path))

    @property
    def help(self):
        return self.eng.help


class GroupBase(object):
    """
    Group data parser, take input from pynipt pipe object
    """
    def __init__(self, pipe, package, step_code, filter_dict=None, mask_path=None):
        self._pipe = pipe
        self.set_package(package)
        self._filter_dset(step_code, filter_dict)
        self.set_brainmask(mask_path)

    def set_brainmask(self, mask_path):
        if mask_path is not None:
            import nibabel as nib
            import numpy as np
            mask_img = nib.load(mask_path).get_data()
            if len(mask_img.shape) != 3:
                raise Exception('Brain mask must be 3D data.')

            self._indices_brain = np.nonzero(mask_img)
        else:
            self._indices_brain = None

    def _filter_dset(self, step_code, filter_dict):
        from pynipt import Bucket
        bucket = Bucket(self._pipe.bucket.path)
        # bucket = self._pipe.bucket
        step = [step for step in bucket.params[1].steps if step_code in step][0]

        if filter_dict is not None:
            if 'ext' is not filter_dict.keys():
                filter_dict['ext'] = 'nii.gz'
        else:
            filter_dict = dict(ext='nii.gz')

        self.dset = bucket(1, pipelines=self._selected_package,
                           steps=step, copy=True, **filter_dict)

    def set_package(self, package):
        self._selected_package = package

    def get_package(self):
        return self._selected_package

    @property
    def list(self):
        return self.dset.df

    def __getitem__(self, item):
        import nibabel as nib
        imgobj = nib.load(self.dset[item].Abspath)
        return imgobj