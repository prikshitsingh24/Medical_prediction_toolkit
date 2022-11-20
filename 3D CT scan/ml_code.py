import os
import keras 
import zipfile
import nibabel as nib
from scipy import ndimage  




def read_nifti_file(filepath):
    scan=nib.load(filepath)
    scan=scan.get_fdata()
    return scan 

