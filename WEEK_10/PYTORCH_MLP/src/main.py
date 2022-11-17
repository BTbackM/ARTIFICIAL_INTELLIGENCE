# NOTE: Importing the libraries

from os import path
from sklearn.model_selection import train_test_split
from utils import parse_csv, DATA_PATH

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn

# NOTE: Importing the dataset

parse_csv('iris.csv')
dataset = pd.read_csv(path.join(DATA_PATH, 'parsed.csv'), header=None)
dataset.columns = ['Sepal L', 
                   'Sepal W', 
                   'Petal L', 
                   'Petal W', 
                   'Species']
print(dataset.head())