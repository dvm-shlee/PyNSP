from pynsp.base import pn


def save_as(img_data, tmpobj, filename):
    nii = pn.ImageObj(img_data, tmpobj.image.affine)
    nii.save_as(filename)