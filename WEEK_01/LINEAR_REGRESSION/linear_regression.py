from numpy.random import seed, rand

def hypothesis(x_i, w: float, b : float) -> float:
    h_xi = x_i  * w + b

    return h_xi

def loss_function(x, y, w : float, b : float) -> float:
    n = len(x)
    l = 1 / (2 * n)

    sum = 0
    for x_i, y_i in zip(x, y):
        sum += pow(y_i - hypothesis(x_i, w, b), 2)

    l *= sum

    return l

def derivate(x, y, w : float, b : float) -> tuple[float, float]:
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


def update(w : float, b : float, dW : float, dB : float, alpha : int) -> tuple[float, float]:
    w = w - alpha * dW
    b = b - alpha * dB

    return w, b

def train(x, y, threshold, alpha) -> tuple[float, float, list]:
    seed(2001)
    w = rand() + 1
    b = rand() + 1
    L = loss_function(x, y, w, b)
    loss = []
    
    while(L > threshold):
        print(L)
        dB, dW = derivate(x, y, w, b)
        w, b = update(w, b, dW, dB, alpha)
        L = loss_function(x, y, w, b)
        loss.append(L)

    return w, b, loss
