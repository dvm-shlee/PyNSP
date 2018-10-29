from __future__ import division
from .base import ImageBase, TimeSeriesBase
import numpy as np


class HandlerBase(object):

    _processed = dict()
    def __init__(self):
        pass

    def __getitem__(self, key):
        return self._processed[key]

    def __setitem__(self, key, item):
        self._processed[key] = item


class ImageHandler(ImageBase, HandlerBase):
    """
    Image handler class
    """

    def __init__(self, img_path):
        super(ImageHandler, self).__init__(img_path)

    @property
    def apply(self):
        return self._apply_func2img

    def set_brainmask(self, mask_path=None):
        """
        This method prepare brain mask indices for processing data

        :param mask_path: path of the mask file
        :return:
        """
        if mask_path is None:
            mask_img = self.img_data
        else:
            mask_img = ImageBase(mask_path).img_data
        dim = len(mask_img.shape)

        if dim is 4:
            mask_img = mask_img.mean(-1)
        elif dim is 3:
            pass
        elif dim is 5:
            mask_img = mask_img.mean(-1).mean(-1)
        else:
            raise Exception     #TODO: Exception message handler
        self._mean_brain = mask_img
        self._indices_brain = np.transpose(np.nonzero(mask_img))

    def _apply_func2img(self, function, *args, **kwargs):
        """
        Apply processing function to image or voxel wise

        :param function:
        :param level: {('timeseries', 'image'), optional}
        :param args: optional arguments
        :param kwargs: optional key:value arguments
        :return:
        """
        key, level = None, 'timeseries'
        for k, item in kwargs.items():
            if k is 'key':
                key = item
                del (kwargs[k])
            if k is 'level':
                level = item
                del (kwargs[k])
        if key is None:
            img_data = self.img_data
        else:
            img_data = self._processed[key]
        # map(np.ndarray.tolist, np.array_split(np.array(indices), 3, axis=0)) <- split indices into 3 parts
        if level is 'timeseries':
            output = np.zeros(self.img_shape)
            print("** {} is applying onto {} voxels..".format(function.__name__,
                                                              len(self._indices_brain)))
            for i, j, k in self._indices_brain:
                ts_data = img_data[i, j, k, :]
                try:
                    output[i, j, k, :] = function(ts_data, *args, **kwargs)
                except:
                    output[i, j, k, :] = np.zeros(ts_data.shape)
        elif level is 'image':
            output = function(img_data, *args, **kwargs)
        else:
            raise Exception         #TODO: Exception message handler

        return np.nan_to_num(output)


class TimeSeriesHandler(TimeSeriesBase, HandlerBase):
    """
    Time series handler class
    """
    def __init__(self, path):
        super(TimeSeriesHandler, self).__init__(path)

    def set_columns(self, columns, key=None):
        if key is None:
            self.df.columns = columns
        else:
            self[key].columns = columns

    def _apply_func2ts(self, function, *args, **kwargs):
        """

        :param function:
        :param level: {'columns', 'dataframe'}
        :param args:
        :param kwargs:
        :return:
        """
        from pandas import DataFrame

        key, level = None, 'dataframe'
        for k, item in kwargs.items():
            if k is 'key':
                key = item
            if k is 'level':
                level = item
            del(kwargs[k])

        output_data = dict()

        if key is None:
            dataframe = self.df
        else:
            dataframe = self[key]

        if level is 'columns':
            for col in dataframe.columns:
                data = dataframe[col]
                output_data[col] = function(data, *args, **kwargs)
            return DataFrame(output_data)
        elif level is 'dataframe':
            return DataFrame(function(dataframe, *args, **kwargs))
        else:
            raise Exception

    @property
    def columns(self):
        return self.df.columns

    @property
    def apply(self):
        return self._apply_func2ts

    def plot(self, *args, **kwargs):
        return self.df.plot(*args, **kwargs)

    def __repr__(self):
        return str(self.df)

    def __iter__(self):
        for col in self.df.columns:
            yield self.df[col]
