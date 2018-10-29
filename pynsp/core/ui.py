from .handler import ImageHandler, TimeSeriesHandler


class RSFC(ImageHandler):
    """
    Processor object to calculate Resting State Functional Connectivity(RSFC) parameters.
    """

    def __init__(self, path, ort_paths=None, mask_path=None, **kwargs):
        super(RSFC, self).__init__(path)
        self._parameter_parsing(**kwargs)
        self.set_brainmask(mask_path)

        if ort_paths is not None:
            if isinstance(ort_paths, str):
                ort_paths = [ort_paths]
            self._orts = self.set_ort(*ort_paths)

    def _parameter_parsing(self, **kwargs):
        """
        Main parameter preparation

        :param kwargs:
        :return:
        """
        self._dt = None
        self._band = None
        self._order = None
        self._orts = None
        self._processed = dict()
        self._indices_brain = None
        self._mean_brain = None

        for k, item in kwargs.items():
            if k is 'dt':
                self._dt = item
            if k is 'band':
                self._band = item
            if k is 'order':
                self._order = item

        if self._dt is None:
            self._dt = self._header['pixdim'][4]
        if self._order is None:
            self._order = 3

    def set_ort(self, *ort_paths):
        """
        Preparing nuisance regressors

        :param ort_paths:
        :return:
        """
        from sklearn.linear_model import BayesianRidge
        from ..methods.signal import polynomial_detrend, standard_norm
        import pandas as pd

        ort_handler = []
        for path in ort_paths:
            orts = TimeSeriesHandler(path)
            orts['Detrended'] = orts.apply(polynomial_detrend,
                                           BayesianRidge,
                                           order=self._order,
                                           level='columns')
            ort_handler.append(orts.apply(standard_norm,
                                          key='Detrended',
                                          level='columns'))
        return pd.concat(ort_handler, axis=1, ignore_index=True)

    def mode_norm(self, **kwargs):
        """
        Mode normalization tool
        :param mode:
        :param kwargs:
        :return:
        """
        from ..methods.signal import mode_norm
        import time
        import numpy as np

        # Default parameters
        mode = 1000
        decimals = 3
        key = None

        for k, item in kwargs.items():
            if k is 'mode':
                mode = item
            if k is 'decimals':
                decimals = item
            if k is 'key':
                key = item

        print('Mode {} normalization...'.format(mode))
        start_time = time.time()
        n_processed = len(self._processed.keys())
        step = '{}.ModeNorm'.format(str(n_processed).zfill(3))
        self[step] = self.apply(mode_norm, self._indices_brain,
                                mode=mode,
                                decimals=decimals, level='image', key=key)
        print("Done...({} sec)".format(np.round(time.time() - start_time, decimals=3)))

    def _denoising(self, orts, **kwargs):
        """
        Core function for applying regression based denoising
        :param orts:
        :param kwargs:
        :return:
        """
        from sklearn.linear_model import BayesianRidge

        from ..methods.signal import nuisance_regression
        import time
        import numpy as np

        # Default parameters
        key = None
        order = self._order

        for k, item in kwargs.items():
            if k is 'key':
                key = item
            if k is 'order':
                order = item

        print('Nuisance_regression...')
        start_time = time.time()
        n_processed = len(self._processed.keys())
        step = '{}.Regressed'.format(str(n_processed).zfill(3))
        self[step] = self.apply(nuisance_regression,
                                BayesianRidge,
                                ort=orts, order=order, level='timeseries', key=key)
        print("Done...({} sec)".format(np.round(time.time() - start_time, decimals=3)))

    def bandpass_filtering(self, **kwargs):
        """
        Apply bandpass filter
        :param kwargs:
        :return:
        """
        from ..methods.signal import freqency_filter
        import time
        import numpy as np

        # Default parameters
        dt = self._dt
        band = self._band
        key = None
        order = 5
        btype = 'band'

        for k, item in kwargs.items():
            if k is 'dt':
                dt = item
            if k is 'band':
                band = item
            if k is 'key':
                key = item
            if k is 'order':
                order = item
            if k is 'btype':
                btype = item

        print("Applying bandpass filter with {}Hz..".format(band))
        start_time = time.time()
        n_processed = len(self._processed.keys())
        step = '{}.Bandpass'.format(str(n_processed).zfill(3))
        self[step] = self.apply(freqency_filter, band, dt=dt,
                                order=order, btype=btype,
                                key=key, level='timeseries')
        print("Done...({} sec)".format(np.round(time.time() - start_time, decimals=3)))

    def nuisance_denoising(self, **kwargs):
        """
        Applying nuisance denoising
        :param kwargs:
        :return:
        """
        self._denoising(self._orts, **kwargs)

    def calc_ReHo(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        import time
        import numpy as np
        from ..methods.rsfc import map_ReHo

        # Default parameters
        key = None
        NN = 3

        for k, item in kwargs.items():
            if k is 'key':
                key = item
            if k in 'NN':
                NN = item
        n_processed = len(self._processed.keys())
        step = '{}.ReHo'.format(str(n_processed).zfill(3))
        print("Calculating ReHo..")
        start_time = time.time()
        self[step] = self.apply(map_ReHo, self._indices_brain,
                                NN=NN, key=key, level='image')
        print("Done...({} sec)".format(np.round(time.time() - start_time, decimals=3)))

    def calc_ROI_CC(self, atlas_path, atlas_label, use_PCA=None, use_Bootstrap=None,
                    fwe="Benjamini-Hochberg", key=None):
        """

        :param atlas_path:
        :param atlas_label:
        :param use_PCA:
        :param fwe:
        :param key:
        :return:
        """
        from ..methods.stats import map_connectivity_with_roi
        from .io import load
        import numpy as np
        import time

        atlas = load(atlas_path).get_data()
        list_rois = map(int, list(set(atlas.flatten()))[1:])
        for roi in list_rois:
            print("Calculating brain-wise connectivity for {}..".format(atlas_label[roi-1]))
            start_time = time.time()
            n_processed = len(self._processed.keys())
            step1 = '{}.{}_Corr'.format(str(n_processed).zfill(3), atlas_label[roi-1])
            step2 = '{}.{}_Pval'.format(str(n_processed+1).zfill(3), atlas_label[roi-1])
            roi_data = (atlas == roi)
            roi_indices = np.transpose(np.nonzero(roi_data))

            r, p = self.apply(map_connectivity_with_roi,
                              roi_indices, self._indices_brain,
                              use_PCA=use_PCA, use_Bootstrap=use_Bootstrap,
                              key=key, level='image')

            print("Done...({} sec)".format(np.round(time.time() - start_time, decimals=3)))
            self[step1] = r
            if fwe is not None:
                print("Correcting Family-Wise Error using {}.".format(fwe))
                from ..methods.stats import multicomp_pval_correction
                self[step2] = multicomp_pval_correction(p, c_type=fwe)
            else:
                self[step2] = p

    def calc_ALFF(self, **kwargs):
        from ..methods.rsfc import map_ALFF
        import numpy as np
        import time
        dt, band, key = self._dt, self._band, None
        for k, item in kwargs.items():
            if k is 'dt':
                dt = item
            if k is 'band':
                band = item
            if k is 'key':
                key = item
        print("Calculating resting-state parameters..")
        start_time = time.time()
        n_processed = len(self._processed.keys())
        step1 = '{}.Pspec'.format(str(n_processed).zfill(3))
        step2 = '{}.ALFF'.format(str(n_processed+1).zfill(3))
        step3 = '{}.fALFF'.format(str(n_processed+2).zfill(3))
        f, Pspec, ALFF, fALFF = self.apply(map_ALFF, self._indices_brain,
                                           dt, band, key=key, level='image')
        print("Done...({} sec)".format(np.round(time.time() - start_time, decimals=3)))
        self['{}_freq'.format(step1)] = f
        self[step1] = Pspec
        self[step2] = ALFF
        self[step3] = fALFF

    @property
    def processed(self):
        return sorted(self._processed.keys())


class QC(ImageHandler, TimeSeriesHandler):
    """
    Processor object to calculate Quality Control parameters.
    """
    def __init__(self, img_path, mparam_path=None, mask_path=None,
                 calc_all=False, **kwargs):
        ImageHandler.__init__(self, img_path)
        TimeSeriesHandler.__init__(self, mparam_path)
        self._processed = dict()
        self.set_brainmask(mask_path=mask_path)
        self.set_columns(['Roll', 'Pitch', 'Yaw', 'dI-S', 'dR-L', 'dA-P'])
        # self._prep_img(**kwargs)
        self._prep_volreg(**kwargs)
        if calc_all is True:
            self.calc_ALL()

    @property
    def apply(self):        # Deactivating apply function
        return None

    @property
    def apply_img(self):
        return self._apply_func2img

    @property
    def apply_ts(self):
        return self._apply_func2ts

    @property
    def mparam(self):
        return self.df

    @property
    def FD(self):
        return self._processed['FD']

    @property
    def DVARS(self):
        return self._processed['DVARS']

    @property
    def VWI(self):
        return self._processed['VWI']

    @property
    def STD(self):
        return self._processed['STD']

    @property
    def tSNR(self):
        return self._processed['tSNR']

    def _prep_img(self, **kwargs):

        mode = None
        decimals = None

        for k, item in kwargs.items():
            if k is 'mode':
                mode = item
            if k is 'order':
                decimals = item
        if mode is not None:
            from ..methods.signal import mode_norm
            self._img_data = self.apply_img(mode_norm,
                                            indices=self._indices_brain,
                                            mode=mode,
                                            decimals=decimals,
                                            level='image')

    def _prep_volreg(self, **kwargs):

        mean_radius = None
        order = 3

        for k, item in kwargs.items():
            if k is 'mean_radius':
                mean_radius = item
            if k is 'order':
                order = item

        if mean_radius is None:
            import numpy as np
            # use distance from aural to central fissure of rat
            r = np.round(np.sqrt(2) * 9)
        else:
            r = mean_radius

        for col in self.columns:
            from ..methods.signal import polynomial_detrend
            from sklearn.linear_model import BayesianRidge

            self.df[col] = polynomial_detrend(self.df[col], BayesianRidge, order)
        from ..methods.qc import convert_radian2distance
        self._dataframe = convert_radian2distance(self.df, r)

    def calc_ALL(self):
        self.calc_FD()
        self.calc_DVARS()
        self.calc_STD()
        self.calc_tSNR()

    def calc_FD(self):
        from ..methods.qc import calc_displacements
        self['FD'] = self._apply_func2ts(calc_displacements, level='dataframe')

    def calc_DVARS(self):
        from ..methods.signal import demean
        from ..methods.qc import calc_BOLD_properties
        self['Demeaned'] = self.apply_img(demean, level='timeseries')
        self['DVARS'], self['VWI'] = self.apply_img(calc_BOLD_properties,
                                                    indices=self.mask,
                                                    level='image',
                                                    key='Demeaned')

    def calc_STD(self):
        from ..methods.qc import map_STD
        self['STD'] = self.apply_img(map_STD,
                                     indices=self.mask,
                                     level='image')

    def calc_tSNR(self):
        from ..methods.qc import map_tSNR
        self['tSNR'] = self.apply_img(map_tSNR,
                                      indices=self.mask,
                                      level='image')

    def plot(self, *args, **kwargs):
        import pandas as pd
        qc_all = [v for k, v in self._processed.items() if isinstance(v, pd.DataFrame)]
        pd.concat(qc_all, axis=1).plot(*args, **kwargs)

    def __repr__(self):
        output = ['********************',
                  'Contents information',
                  '********************',
                  'Data in [{}.mparam (mm)] : Six rigid motion parameters\n'.format(self.__class__.__name__),
                  'Data in [{}.FD]'.format(self.__class__.__name__),
                  '- FD   (mm)    : Framewise Displacement',
                  '- ATD  (mm)    : Absolute Translational Displacement',
                  '- ARD  (mm)    : Absolute Rotational Displacement\n',
                  'Data in [{}.DVARS]'.format(self.__class__.__name__),
                  '- DVgs (BOLD)  : DVARS of global signal in brain mask',
                  '- SDgs (BOLD)  : Standard deviation of signal in brain mask',
                  '- GS   (BOLD)  : Demeaned global signal in brain mask\n',
                  'Data of [{}.VWI] : Voxel-wise intensities fluctuation (2D)'.format(self.__class__.__name__),
                  'Data of [{}.STD] : Brain-wise Standard Deviation(STD) (3D)'.format(self.__class__.__name__),
                  'Data of [{}.STD] : Brain-wise temporal Signal to Noise ratio (tSNR) (3D)'.format(self.__class__.__name__)
                  ]
        return '\n'.join(output)

    def __getitem__(self, key):
        return self._processed[key]

    def __setitem__(self, key, item):
        self._processed[key] = item