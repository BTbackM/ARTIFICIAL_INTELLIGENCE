from numpy import amin, amax
from numpy.linalg import norm

# NOTE: This function requires an array to be normalized
def normalize(array):
    min_x, max_x = amin(array), amax(array)

    for i in range(len(array)):
        array[i] = (array[i] - min_x) / (max_x - min_x)
