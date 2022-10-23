from sklearn.neighbors import KDTree

import numpy as np

class KNN():
    def __init__(self):
        pass

    @staticmethod
    def train(X):
        tree = KDTree(X, leaf_size = 10)

        return tree

    @staticmethod
    def search(tree, X, k):
        indexes = tree.query(X, k, return_distance = False)

        return indexes

    @staticmethod
    def vote(tree, X, Y, k):
        max_keys = []
        indexes = KNN.search(tree, X, k)

        for index in indexes:
            key, value = np.unique(Y[index], return_counts = True)
            mv = np.where(value == np.amax(value))[0]
            if mv.shape[0] == 1:
                max_keys.append(key[mv])
            else:
                max_keys.append(None)

        return np.array(max_keys)
