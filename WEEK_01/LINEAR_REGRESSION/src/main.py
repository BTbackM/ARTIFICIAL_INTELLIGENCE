import numpy as np
import pandas as pd

from linear_regression import train
from os import path
from sklearn.linear_model import LinearRegression
from utils import delete_images, normalize, make_gif, make_scatter
from utils import DATA_PATH, IMG_PATH, blue

df = pd.DataFrame(pd.read_csv(path.join(DATA_PATH, 'db.csv')))

x = np.array(df [['income']])
y = np.array(df [['happiness']])

normalize(x)
normalize(y)

make_scatter(x, y, blue, 'Data', 'Income', 'Happiness', 'Linear Regression', path.join(IMG_PATH, 'data.jpg'))

model = LinearRegression().fit(x, y)
w, b = model.coef_, model.intercept_

w, b, loss = train(x, y, 0.1, 10 ** -2, w, b)

make_gif(path.join(IMG_PATH, 'linear_regression/*.jpg'), path.join(IMG_PATH, 'linear_regression.gif'))
delete_images(path.join(IMG_PATH, 'linear_regression/'))
