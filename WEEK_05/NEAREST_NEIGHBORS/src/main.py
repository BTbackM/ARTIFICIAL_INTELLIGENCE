from knn_nearest import KNN
from utils import read_images
from utils import DATA_PATH

import os
import numpy as np

train = os.path.join(DATA_PATH, 'images/test_train')
validation = os.path.join(DATA_PATH, 'images/test_validation')

Xt, Yt = read_images(train)
Xv, Yv = read_images(validation)

tree = KNN.train(Xt)
max_keys = KNN.vote(tree, Xv, Yt, 15)

print(max_keys.shape)
print(np.unique(np.equal(max_keys.T, Yv.T), return_counts = True))
