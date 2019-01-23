import numpy as np
import skimage
import sys
from skimage.io import *
from os import walk

imgs_dir_path = sys.argv[1]
img_path = sys.argv[2]

def scaling(M):
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

def reconstruct(img, mean, components):
    x = img - mean
    tmp = []
    for u in components:
        result = np.dot(np.dot(u, x), u)
        tmp.append(result)
    tmp = np.array(tmp)
    return (np.sum(tmp, axis=0) + mean).reshape(600, 600, 3)


if __name__ == '__main__':
    f = []
    for (dirpath, dirname, filenames) in walk(imgs_dir_path):
        f.extend(filenames)
    imgs = []
    for filename in f:
        filepath = imgs_dir_path + '/' + filename
        imgs.append(imread(filepath).reshape(600*600*3))
        
    imgs = np.array(imgs)
    mean = np.mean(imgs, axis=0)
    X = imgs - mean
    M = np.dot(X, X.T)  # compact trick
    e, V = np.linalg.eig(M)
    tmp = np.dot(X.T, V).T
    S = np.nan_to_num(np.sqrt(e))
    U = [tmp[i]/S[i] for i in range(4)]

    img = imread(img_path).reshape(600*600*3)
    reconstruction = reconstruct(img, mean, U)
    imsave('reconstruction.jpg', scaling(reconstruction))
