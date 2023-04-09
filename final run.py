import zipfile  
dataset_path = "BraTS_2018_Data_Training.zip"  #downloaded on ada just check directory and run
zfile = zipfile.ZipFile(dataset_path)
zfile.extractall()
import SimpleITK as sitk  
import numpy as np  
from model import build_model  
import glob  
from scipy.ndimage import zoom 
import re  

def read_img(img_path):
   
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def resize(img, shape, mode='constant', orig_shape=(155, 240, 240)):
    
    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        shape[0]/orig_shape[0],
        shape[1]/orig_shape[1], 
        shape[2]/orig_shape[2]
    )
    
    
    return zoom(img, factors, mode=mode)


def preprocess(img, out_shape=None):
   
    if out_shape is not None:
        img = resize(img, out_shape, mode='constant')
    
    
    mean = img.mean()
    std = img.std()
    return (img - mean) / std


def preprocess_label(img, out_shape=None, mode='nearest'):
    
    ncr = img == 1  # (NCR/NET)
    ed = img == 2  #  (ED)
    et = img == 4  #  (ET)
    
    if out_shape is not None:
        ncr = resize(ncr, out_shape, mode=mode)
        ed = resize(ed, out_shape, mode=mode)
        et = resize(et, out_shape, mode=mode)

    return np.array([ncr, ed, et], dtype=np.uint8)

#data loading ask viraj to verify

t1 = glob.glob('*GG/*/*t1.nii.gz')
t2 = glob.glob('*GG/*/*t2.nii.gz')
flair = glob.glob('*GG/*/*flair.nii.gz')
t1ce = glob.glob('*GG/*/*t1ce.nii.gz')
seg = glob.glob('*GG/*/*seg.nii.gz')  

pat = re.compile('.*_(\w*)\.nii\.gz')

data_paths = [{
    pat.findall(item)[0]:item
    for item in items
}
for items in list(zip(t1, t2, t1ce, flair, seg))]

input_shape = (4, 80, 96, 64)
output_channels = 3
data = np.empty((len(data_paths[:4]),) + input_shape, dtype=np.float32)
labels = np.empty((len(data_paths[:4]), output_channels) + input_shape[1:], dtype=np.uint8)

import math

total = len(data_paths[:4])
step = 25 / total

for i, imgs in enumerate(data_paths[:4]):
    try:
        data[i] = np.array([preprocess(read_img(imgs[m]), input_shape[1:]) for m in ['t1', 't2', 't1ce', 'flair']], dtype=np.float32)
        labels[i] = preprocess_label(read_img(imgs['seg']), input_shape[1:])[None, ...]
        
        
        print('\r' + f'Update: '
            f"[{'=' * int((i+1) * step) + ' ' * (24 - int((i+1) * step))}]"
            f"({math.ceil((i+1) * 100 / (total))} %)",
            end='')
    except Exception as e:
        print(f'Error with {imgs["t1"]}, skipping...\n Exception:\n{str(e)}')
        continue

model = build_model(input_shape=input_shape, output_channels=3)

model.fit(data, [labels, data], batch_size=1, epochs=1)