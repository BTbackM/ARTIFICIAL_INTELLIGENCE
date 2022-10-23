from numpy.random import seed, rand
from os import path
from utils import IMG_PATH, GREEN
from utils import redo_figures

def hypothesis(x_i, w, b):
    h_xi = (x_i * w) + b

    return h_xi

def loss_function(x, y, w, b):
    n = len(x)
    l = 1 / (2 * n)

    S = 0
    for x_i, y_i in zip(x, y):
        e = (y_i - hypothesis(x_i, w, b))
        S += pow(e, 2)
    # S = sum(map(lambda x_i, y_i: (y_i - hypothesis(x_i, w, b)) ** (y_i - hypothesis(x_i, w, b)), x, y))

    l *= S

    return l

def derivate(x, y, w, b):
    n = len(x)
    dW = (1 / n)
    dB = (1 / n)
    
    # S_w = sum(map(lambda x_i, y_i: (y_i - hypothesis(x_i, w, b)) * -x_i, x, y))
    # S_b = sum(map(lambda x_i, y_i: (y_i - hypothesis(x_i, w, b)) * -1, x, y))
    S_w, S_b = 0, 0
    for x_i, y_i in zip(x, y):
        tmp = (y_i - hypothesis(x_i, w, b))
        S_w += (tmp * -x_i)
        S_b += (tmp * -1)

    dW *= S_w
    dB *= S_b

    return dW, dB


def update(w, b, dW, dB, alpha):
    w = w - alpha * dW
    b = b - alpha * dB

    return w, b

def train(x, y, epochs, alpha, _w, _b):
    seed(2001)
    w = 0
    b = 1
    L = loss_function(x, y, w, b)
    loss = []
    frame = 0
    
    for i in range(epochs):
        if i % 50 == 0:
            frame_number = '{:0>4d}'.format(frame)
            frame += 1
            redo_figures(x, y, w, b, _w, _b, path.join(IMG_PATH, f'linear_regression/linear_regression_{frame_number}.jpg'))
        dW, dB = derivate(x, y, w, b)
        w, b = update(w, b, dW, dB, alpha)
        L = loss_function(x, y, w, b)
        loss.append(L)

    redo_figures(x, y, w, b, _w, _b, path.join(IMG_PATH, f'linear_regression.jpg'))

    return w, b, loss
