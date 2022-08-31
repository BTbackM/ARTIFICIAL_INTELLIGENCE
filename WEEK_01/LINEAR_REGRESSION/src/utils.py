from glob import glob
from matplotlib.pyplot import legend, plot, savefig, scatter, title, xlabel, ylabel, xlim, ylim
from numpy import amin, amax
from os import listdir, path, remove
from PIL import Image

# ------------------------ VARIABLES ------------------------

# ----- COLORS -----
blue = '#3264AF'
green = '#197D32'
red = '#AF3232'

ABS_PATH = path.dirname(path.realpath(__file__))
DATA_PATH = path.join(ABS_PATH, '../data/')
IMG_PATH = path.join(ABS_PATH, '../img/')

# ------------------------ FUNCTIONS ------------------------

# NOTE: This function requires an array to be normalized
def normalize(array):
    min_x, max_x = amin(array), amax(array)

    for i in range(len(array)):
        array[i] = (array[i] - min_x) / (max_x - min_x)

def make_gif(FRAMES_PATH, GIF_PATH):
    frames = [Image.open(img) for img in sorted(glob(FRAMES_PATH))]
    
    frame_one = frames[0]
    frame_one.save(GIF_PATH, format = 'GIF', append_images = frames, save_all = True, duration = 250, loop = 0)

def make_scatter(x, y, COLOR, LABEL, XLABEL, YLABEL, TITLE, PATH):
    scatter(x, y, color = COLOR, marker = 'o', s = 20, label = LABEL)
    xlabel(XLABEL, fontweight = 'bold')
    ylabel(YLABEL, fontweight = 'bold')
    title(TITLE.upper(), fontsize = 14, fontweight = 'bold')
    legend(loc=2)
    xlim([0, 1])
    ylim([0, 2])
    savefig(PATH, dpi = 250)

def make_plot(x, y, COLOR, LABEL, XLABEL, YLABEL, TITLE, PATH):
    plot(x, y, color = COLOR, label = LABEL, linewidth = 3)
    xlabel(XLABEL, fontweight = 'bold')
    ylabel(YLABEL, fontweight = 'bold')
    title(TITLE.upper(), fontsize = 14, fontweight = 'bold')
    legend(loc=2)
    xlim([0, 1])
    ylim([0, 2])
    savefig(PATH, dpi = 250)

def delete_images(directory):
    for file in listdir(directory):
        remove(path.join(directory, file))
