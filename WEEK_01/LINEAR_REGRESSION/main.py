import numpy as np
import pandas as pd

from global_functions import normalize
from linear_regression import train
from matplotlib.pyplot import plot, savefig
from sklearn.linear_model import LinearRegression

df = pd.DataFrame(pd.read_csv('./db.csv'))

x = np.array(df [['income']])
y = np.array(df [['happiness']])

normalize(x)
normalize(y)

plot(x, y, '.')
savefig('./data.jpg', dpi = 250)

w, b, loss = train(x, y, 0.1, 10 ** -4)

plot(x, w * x + b)
savefig('./linear_regression.jpg', dpi = 250)

model = LinearRegression().fit(x, y)
w, b = model.coef_, model.intercept_

plot(x, w * x + b)
savefig('./linear_regression.jpg', dpi = 250)
