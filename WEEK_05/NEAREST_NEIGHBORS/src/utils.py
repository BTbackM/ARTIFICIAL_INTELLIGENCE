from skimage.io import imread

import numpy as np
import os
import pywt
import pywt.data

# ------------------------ VARIABLES ------------------------

# ----- COLORS -----

BLUE = '#3264AF'
GREEN = '#197D32'
RED = '#AF3232'

ABS_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ABS_PATH, '../data/')
IMG_PATH = os.path.join(ABS_PATH, '../img/')

# ------------------------ FUNCTIONS ------------------------

@staticmethod
def haar(img, slices):
    for _ in range(slices):
        img, _ = pywt.dwt2(img, 'haar')

    return img

@staticmethod
def vectorize(matrix):
    return matrix.flatten()

@staticmethod
def preprocess(img):
    return vectorize(haar(img, 2))

@staticmethod
def read_images(path):
    X = []
    Y = []

    for dir in os.listdir(path):
        d = os.path.join(path, dir)
        if os.path.isdir(d):
            for file in os.listdir(d):
                img = os.path.join(d, file)
                img = imread(img)
                X.append(preprocess(img))
                Y.append(dir)

    expr = np.unique(Y)
    expr = dict(zip(expr, range(len(expr))))
    Y = np.array(list(map(lambda e: expr[e], Y)))
    X = np.array(X)
    Y = np.array(Y)

    return X, Y
