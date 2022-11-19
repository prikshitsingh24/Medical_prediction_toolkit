import os
import keras 
import zipfile
import nibabel as nib
from scipy import ndimage 


url=""
filename=os.path.join(os.gatcwd(),"CT-0.zip")
keras.utils.get_file(filename,url)

url=""
filename=os.path.join(os.gatcwd(),"CT-23.zip")
keras.utils.get_file(filename,url)

os.makedirs("MosmedData")

with zipfile.ZipFile("CT-0.zip","r") as z_fp:
    z_fp.extractall("./MosMedData/")

with zipfile.ZipFile("CT-23.zip","r") as z_fp:
    z_fp.extractall("./MosMedData/")


def read_nifti_file(filepath):
    scan=nib.load(filepath)
    scan=scan.get_fdata()
    return scan 

