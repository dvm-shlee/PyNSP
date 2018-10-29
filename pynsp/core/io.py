def load(filename):
    """
    load available file
    available exts: .nii(.gz), .mha, .xls(x), .csv, .tsv, .json

    :param filename: file want to load
    :type filename: str
    :return: object
    """
    if '.nii' in filename:
        from nibabel import Nifti1Image
        img = Nifti1Image.load(filename)
    else:
        import pandas as pd
        if '.xls' in filename:
            img = pd.read_excel(filename)
        elif '.csv' in filename:
            img = pd.read_csv(filename)
        elif '.tsv' in filename:
            img = pd.read_table(filename)
        elif '.1D' in filename:
            img = pd.read_csv(filename, header=None, sep='\s+')
        elif '.json' in filename:
            import json
            img = json.load(open(filename))
        else:
            raise Exception #TODO: Exception message handler
    return img


def save(data_obj, filename, key=None):
    """
    Save file

    :param filename: filename
    """
    if hasattr(data_obj, '_img_data'):
        if not hasattr(data_obj, 'df'):
            from nibabel import Nifti1Image
            if key is None:
                data = data_obj._img_data
            else:
                data = data_obj._processed[key]
            nii = Nifti1Image(data, data_obj._affine)
            nii._header = data_obj._header
            if filename.endswith('.nii.gz'):
                output_path = filename
            else:
                output_path = '{}{}'.format(filename, '.nii.gz')
            nii.to_filename(output_path)
        else:
            pass
    elif hasattr(data_obj, 'df'):
        if key is None:
            data = data_obj.df
        else:
            data = data_obj._processed[key]
        if filename.endswith('.xlsx'):
            output_path = filename
        else:
            output_path = '{}{}'.format(filename, '.xlsx')
        data.to_excel(output_path, index=False)
    else:
        raise Exception  #TODO: Exception message handler