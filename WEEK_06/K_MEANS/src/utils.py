from glob import glob
from os import listdir, path, remove
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np

# ------------------------ VARIABLES ------------------------

# ----- COLORS -----

BLUE = '#3264B0'
GREEN = '#197D30'
RED = '#AF3230'
ORANGE = '#DD8120'
PURPLE = '#490160'
SKY = '#32BBAA'
GRAY = '#6B6B6B'
LR_COLORS = [BLUE, GREEN, ORANGE, RED, SKY, PURPLE, GRAY]

ABS_PATH = path.dirname(path.realpath(__file__))
DATA_PATH = path.join(ABS_PATH, '../data/')
IMG_PATH = path.join(ABS_PATH, '../img/')
DPI = 250
LR = np.array([10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3, 10 ** -4, 2 * 10 ** -5, 5 * 10 ** -5])
TARGET_NAMES = [' A ', ' B ', ' C ', ' D ', ' E ', ' F ',
        ' G ', ' H ', ' I ', ' K ', ' L ', ' M ',
        ' N ', ' O ', ' P ', ' Q ', ' R ', ' S ',
        ' T ', ' U ', ' V ', ' W ', ' X ', ' Y ']

# ------------------------ FUNCTIONS ------------------------

def delete_images(directory):
    for file in listdir(directory):
        remove(path.join(directory, file))

def format(df):
    day_to_num={'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6}
    df['day'] = df['day'].map(day_to_num)

    month_to_num={'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
            'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11}
    df['month'] = df['month'].map(month_to_num)

    return df

# NOTE: This function requires an array to be normalized
def normalize(array):
    min_x, max_x = np.amin(array), np.amax(array)

    for i in range(len(array)):
        array[i] = (array[i] - min_x) / (max_x - min_x)

def make_boxplot(data, COLOR, XLABEL, TITLE, PATH, CLF, YLIM = 0.0):
    bp = plt.boxplot(
            x = data,
            labels = XLABEL,
            showfliers = False
            )
    plt.title(TITLE, fontsize = 12, fontweight = 'bold')
    plt.tight_layout()
    if YLIM:
        plt.ylim([0, YLIM])

    for median, color in zip(bp['medians'], COLOR):
        median.set(color = color, linewidth = 2)
    
    plt.savefig(PATH, dpi = DPI)
    if CLF:
        plt.clf()

def make_confussion_matrix(Y_true, Y_pred, target_names, PATH):
    _, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlabel('', fontweight = 'bold')
    ax.set_ylabel('' ,fontweight = 'bold')

    ConfusionMatrixDisplay.from_predictions(
            y_true = Y_true,
            y_pred = Y_pred,
            display_labels = target_names,
            normalize = 'true',
            values_format = '.2f',
            cmap = 'Blues',
            ax = ax,
            )

    plt.tight_layout()
    plt.savefig(PATH, dpi = DPI, bbox_inches='tight')
    plt.clf()
