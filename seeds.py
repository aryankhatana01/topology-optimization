import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("h5ds/dataset.h5", 'r') as h5f:
    X = h5f['iters']
    Y = h5f['targets']
    
    print(X[0].shape, Y[0].shape)
    # plt.imshow()
    img = X[1][:, :, 1]
    print(img.shape)
    plt.imshow(img)
    plt.show()
    tar = Y[1]
    plt.imshow(tar)
    plt.show()