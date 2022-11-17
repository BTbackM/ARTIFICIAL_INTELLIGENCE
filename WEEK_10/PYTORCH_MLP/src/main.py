# NOTE: Importing the libraries

from os import path
from sklearn.model_selection import train_test_split
from utils import parse_csv, DATA_PATH
from mlp import Model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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

# NOTE: Splitting the dataset into the training and test set

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# NOTE: Create the model

if torch.cuda.is_available(): 
  dev = 'cuda:0'
else:
  dev = 'cpu'

model = Model(4, 20, 3).to(dev)
print(model)

# NOTE: Create tensors from the data

X_train = torch.FloatTensor(X_train).to(dev)
X_test = torch.FloatTensor(X_test).to(dev)
Y_train = torch.LongTensor(Y_train).to(dev)
Y_test = torch.LongTensor(Y_test).to(dev)

# NOTE: Define the loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):
  Y_pred = model.forward(X_train)
  loss = criterion(Y_pred, Y_train)
  losses.append(loss)
  if i%10 == 0:
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()