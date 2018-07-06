from pynsp.base import pn


def save_as(img_data, tmpobj, filename, temp=True):
    if temp == True:
        affine = tmpobj.image.affine
    else:
        affine = tmpobj.affine
    nii = pn.ImageObj(img_data, affine)
    nii.save_as(filename)
