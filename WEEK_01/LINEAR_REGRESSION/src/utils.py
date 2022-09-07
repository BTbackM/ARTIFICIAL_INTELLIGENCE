from glob import glob
from numpy import amin, amax
from os import listdir, path, remove
from PIL import Image

import matplotlib.pyplot as plt

# ------------------------ VARIABLES ------------------------

# ----- COLORS -----

BLUE = '#3264AF'
GREEN = '#197D32'
RED = '#AF3232'

ABS_PATH = path.dirname(path.realpath(__file__))
DATA_PATH = path.join(ABS_PATH, '../data/')
IMG_PATH = path.join(ABS_PATH, '../img/')

# ------------------------ FUNCTIONS ------------------------

def delete_images(directory):
    for file in listdir(directory):
        remove(path.join(directory, file))

# NOTE: This function requires an array to be normalized
def normalize(array):
    min_x, max_x = amin(array), amax(array)

    for i in range(len(array)):
        array[i] = (array[i] - min_x) / (max_x - min_x)

def make_gif(FRAMES_PATH, GIF_PATH):
    frames = [Image.open(img) for img in sorted(glob(FRAMES_PATH))]
    
    frame_one = frames[0]
    frame_one.save(GIF_PATH, format = 'GIF', append_images = frames, save_all = True, duration = 250, loop = 0)

def make_plot(x, y, COLOR, LABEL, XLABEL, YLABEL, TITLE, PATH):
    plt.plot(x, y, color = COLOR, label = LABEL, linewidth = 3)
    plt.xlabel(XLABEL, fontweight = 'bold')
    plt.ylabel(YLABEL, fontweight = 'bold')
    plt.title(TITLE.upper(), fontsize = 14, fontweight = 'bold')
    plt.legend(loc=2)
    plt.xlim([0, 1])
    plt.ylim([0, 2])
    plt.savefig(PATH, dpi = 250)

def make_scatter(x, y, COLOR, LABEL, XLABEL, YLABEL, TITLE, PATH):
    plt.scatter(x, y, color = COLOR, marker = 'o', s = 20, label = LABEL)
    plt.xlabel(XLABEL, fontweight = 'bold')
    plt.ylabel(YLABEL, fontweight = 'bold')
    plt.title(TITLE.upper(), fontsize = 14, fontweight = 'bold')
    plt.legend(loc=2)
    plt.xlim([0, 1])
    plt.ylim([0, 2])
    plt.savefig(PATH, dpi = 250)

def redo_figures(x, y, w, b, _w, _b, f_path):
    # NOTE: Redo previous plots
    plt.clf()
    make_scatter(x, y, BLUE, 'Data', 'Income', 'Happiness', 'Linear Regression', path.join(IMG_PATH, 'data.jpg'))
    make_plot(x, _w * x + _b, RED, 'Sklearn', 'Income', 'Happiness', 'Linear Regression', path.join(IMG_PATH, 'sklearn.jpg'))
    make_plot(x, w * x + b, GREEN, 'Ours', 'Income', 'Happiness', 'Linear Regression', f_path)

