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
        sum_b += (tmp * -1)

    dW *= sum_w
    dB *= sum_b

    return dW, dB


def update(w, b, dW, dB, alpha):
    w = w - alpha * dW
    b = b - alpha * dB

    return w, b

def train(x, y, epochs, alpha, _w, _b):
    seed(2001)
    w = 0
    b = 0
    L = loss_function(x, y, w, b)
    loss = []
    frame = 0
    
    for _ in range(epochs):
        frame_number = '{:0>4d}'.format(frame)
        frame += 1
        # redo_figures(x, y, w, b, _w, _b, path.join(IMG_PATH, f'linear_regression/linear_regression_{frame_number}.jpg'))
        n = len(x)
        Y_pred = w*x + b  # The current predicted value of Y
        D_w = (-2/n) * sum(x * (y - Y_pred))  # Derivative wrt m
        D_b = (-2/n) * sum(y - Y_pred)  # Derivative wrt c
        w = w - alpha * D_w  # Update m
        b = b - alpha * D_b  # Update c
        # dW, dB = derivate(x, y, w, b)
        # w, b = update(w, b, dW, dB, alpha)
        # L = loss_function(x, y, w, b)
        # loss.append(L)

    redo_figures(x, y, w, b, _w, _b, path.join(IMG_PATH, f'linear_regression.jpg'))

    return w, b, loss
