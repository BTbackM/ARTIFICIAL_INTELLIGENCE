from utils import read_images
from utils import DATA_PATH

import os
import numpy as np

root = os.path.join(DATA_PATH, 'images/train')

X, Y = read_images(root)
print(np.unique(Y, return_counts = True))
