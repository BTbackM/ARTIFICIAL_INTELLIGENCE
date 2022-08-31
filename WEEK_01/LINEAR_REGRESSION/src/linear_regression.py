from matplotlib.pyplot import clf
from numpy.random import seed, rand
from os import path
from utils import IMG_PATH, blue, green, red
from utils import make_plot, make_scatter

def hypothesis(x_i, w, b):
    h_xi = (x_i * w) + b

    return h_xi

def loss_function(x, y, w, b):
    n = len(x)
    l = 1 / (2 * n)

    sum = 0
    for x_i, y_i in zip(x, y):
        sum += pow(y_i - hypothesis(x_i, w, b), 2)

    l *= sum

    return l

def derivate(x, y, w, b):
    n = len(x)
    dW = (1 / n)
    dB = (1 / n)
    
    sum_w, sum_b = 0, 0
    for x_i, y_i in zip(x, y):
        tmp = (y_i - hypothesis(x_i, w, b))
        sum_w += (tmp * -x_i)
        sum_w += (tmp * -1)

    dW *= sum_w
    dB *= sum_b

    return dW, dB


def update(w, b, dW, dB, alpha):
    w = w - alpha * dW
    b = b - alpha * dB

    return w, b

def train(x, y, threshold, alpha, _w, _b):
    seed(2001)
    w = rand() + 1
    b = rand() + 1
    L = loss_function(x, y, w, b)
    loss = []
    frame = 0
    
    while(L > threshold):
        # NOTE: Redo previous plots
        clf()
        frame_number = '{:0>3d}'.format(frame)
        make_scatter(x, y, blue, 'Data', 'Income', 'Happiness', 'Linear Regression', path.join(IMG_PATH, 'data.jpg'))
        make_plot(x, _w * x + _b, red, 'Sklearn', 'Income', 'Happiness', 'Linear Regression', path.join(IMG_PATH, 'sklearn.jpg'))
        make_plot(x, w * x + b, green, 'Ours', 'Income', 'Happiness', 'Linear Regression', path.join(IMG_PATH, f'linear_regression/linear_regression_{frame_number}.jpg'))

        dB, dW = derivate(x, y, w, b)
        w, b = update(w, b, dW, dB, alpha)
        L = loss_function(x, y, w, b)
        loss.append(L)
        frame += 1

    return w, b, loss
