from __future__ import print_function
import os 
import h5py
import numpy as np
from tqdm import tqdm

IMAGE_H, IMAGE_W = 40, 40
N_ITERS = 100

files = os.listdir("TOP4040")
iters_shape = (len(files), IMAGE_H, IMAGE_W, N_ITERS)
iters_chunk_shape = (1, IMAGE_H, IMAGE_W, 1)
target_shape = (len(files), IMAGE_H, IMAGE_W, 1)
target_chunk_shape = (1, IMAGE_H, IMAGE_W, 1)


with h5py.File("h5ds/dataset.h5", 'w') as h5f:
    iters = h5f.create_dataset('iters', iters_shape, chunks=iters_chunk_shape)
    targets = h5f.create_dataset('targets', target_shape, chunks=target_chunk_shape)
    
    for i, file_name in tqdm(enumerate(files), total=len(files)):
        file_path = os.path.join("TOP4040", file_name)
        arr = np.load(file_path)['arr_0']
        arr = arr.transpose((1, 2, 0))
        iters[i] = arr
        
        th_ = arr.mean(axis=(0, 1), keepdims=True)
        targets[i] = (arr > th_).astype('float32')[:, :, [-1]]