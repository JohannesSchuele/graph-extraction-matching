import sys
import os
import datetime
import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from typing import *
from pytorch3d.io import load_objs_as_meshes, save_obj
from cycler import cycler
#plt.style.use('paper_style.mplstyle')

import numpy as np
from skimage import img_as_ubyte
from matplotlib.figure import Figure



SAVE_RESULTS_BOOLEAN = False
AXIS_OFF = True
PLOT_MATCHES = False
PLOT_WORLD_MAP = False # plot world map in 2D and 3D
PLOT_FIRST_IMAGES = False
PLOT_CALIBRATION_RANSAC = False
PLOT_CAMERA_LOSS_ON_MESH = False
PLOT_LOSS_CAMERA_POSE = True
SAVE_LOSS_CAMERA_POSE = True
MAKE_GIF_CAMERA_POSE_ERROR = False
PLOT_PERIOD_POSE_OPT= 5
MAKE_GIF_WORLD_MAP_DEFORMATION = False
MAKE_GIF_WORLD_MAP_DEFORMATION_PERIOD = 20
PLOT_POINT_CLOUD_ALIGNMENT = False
PLOT_PERIOD_POINT_CLOUD_ALIGNMENT = 50
PLOT_PERIOD_DEFORMATION = 25

PLOT_CHAMFER_LOSS_PC_ALIGNMENT = True
PLOT_PERIOD_TEXTURE = 1 #also given to the GifMaker class
PLOT_TEXTURE_COMPARE_AND_MESH = False
# pose optimization with target image
PLOT_PREDICTED_TO_TARGET_IMAGE_POSE_OPTIMIZATION = True
PLOT_PREDICTED_TO_TARGET_IMAGE_POSE_OPTIMIZATION_PERIOD = 20

# plots rendering:
PLOT_STITICHING_CAM2MASH_BLUR_FACTOR = False
PLOT_STITICHING_CAM2MASH_MERGED_POINTS = False #ToDo: reprojection for plotting
PLOT_MASKED_IMAGE_FOR_TESTURE_USAGE = True

# deformation
PLOT_KEY_POINT_CLOUD_ON_MESH = True
PLOT_TARGET_POINT_CLOUD_ON_MESH = True

#RANSAC:
PRINT_RANSAC_REPORJECTION_ERROR = False

PLOT_RE_RENDERED_KEY_POINT_CLOUD_ON_MESH = False

SCALE_FIGURE_SETTINGs = 1.0

#deformation
PLOT_PREDICTED_TO_TARGET_IMAGE_DEFORMATION = True
plot_period_image_deformation = 4

sys.path.append(os.path.abspath(''))
time_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if os.getcwd().__len__() == 69 or os.getcwd().__len__() == 36 or os.getcwd().__len__() == 39: # to get it working from the examples folder
    PLOT_RESULTS_DIR = '../../results/' + time_tag + '/'
else: # by default from the main project level
    PLOT_RESULTS_DIR = '../results/'+time_tag+'/'
file_format = '.pdf'
file_format = '.png'
GIF_DIR = PLOT_RESULTS_DIR+'gifs/'
dpi_resolution = 400

if SAVE_RESULTS_BOOLEAN:

    os.mkdir(PLOT_RESULTS_DIR)
    os.mkdir(GIF_DIR)


def save_figure(fig, name_of_figure, ax= None, show_fig=False, size_width=None, size_height=None, do_tight_fit=True):

    if size_width is not None and size_height is not None:
        size_fig(fig=fig, size_width=size_width, size_height=size_height)

    if AXIS_OFF and ax is not None:
        ax.axis('off')

    if do_tight_fit:
        fig.tight_layout()
    if show_fig:
        fig.show()
    if SAVE_RESULTS_BOOLEAN:
        fp = PLOT_RESULTS_DIR + name_of_figure + file_format
        fig.savefig(fp)
    else:
        print('If the figure is supposed to be saved,\\'
              ' set the variable PLOT_RESULTS_BOOLEAN in config_plots.py!')

def save_mesh(mesh, name_of_mesh, normalize_and_center = False):


    if SAVE_RESULTS_BOOLEAN:
        # Fetch the verts and faces of the final predicted mesh
        final_verts, final_faces = mesh.get_mesh_verts_faces(0)

        if normalize_and_center:
            # Scale normalize back to the original target size
            verts = mesh.verts_packed()
            N = verts.shape[0]
            center = verts.mean(0)
            scale = max((verts - center).abs().max(0)[0])
            final_verts = final_verts * scale + center


        file_format = '.obj'
        Mesh_DIR = PLOT_RESULTS_DIR + 'mesh_reconstructions/'
        dpi_resolution = 400
        os.mkdir(Mesh_DIR)
        #os.mkdir(GIF_DIR)
        final_obj_fp = Mesh_DIR + name_of_mesh + file_format
        # final_obj = os.path.join('./', 'final_baldder_reconstruction_model.obj')
        save_obj(final_obj_fp, final_verts, final_faces)



def generate_mpl_3D_figure(size_width=3.25, size_height=3.25, do_tight_fit=True) -> Tuple[plt.Figure, plt.Axes]:
    # 3D network plot
#sudo apt install texlive-latex-base
    # /graphics/scratch/schuelej/system/miniconda3/envs/pytorch3d/lib/python3.8/site-packages/matplotlib/mpl-data/stylelib
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    # plt.style.use('ggplot') # for prasentations
    #plt.style.use('classic')
    # plt.style.use(['ggplot', 'paper_style'])
    # plt.style.use('dark_background')
    plt.style.use('paper_style')
    #plt.style.use('paper_style_7inch')

    #fig = plt.figure(figsize=(10, 7))
    #ax = fig.get_axes()
    if do_tight_fit:
        fig = plt.figure(tight_layout=True, figsize=(size_width, size_height))
    else:
        fig = plt.figure(tight_layout=False, figsize=(size_width, size_height))

    #ax = fig.gca(projection="3d")
    #fig = size_fig(fig, size_width=3.25, size_height=3.25)
    ax = fig.add_subplot(111, projection='3d')
    # if size_width is not None and size_height is not None:
    #     fig = size_fig(fig, size_width, size_height)


    return fig, ax

def size_fig(fig, size_width=3.25, size_height=3.25):
    #fig.tight_layout()
    fig.set_size_inches(size_width, size_height)
    fig.tight_layout()
    return fig

def show_fig(fig, size_width=None, size_height=None) -> plt.Figure:
    if size_width is not None and size_height is not None:
        fig = size_fig(fig, size_width, size_height)
    fig.tight_layout()
    fig.show()
    return fig

def generate_mpl_figure(size_width=3.25, size_height=3.25) ->Tuple[plt.Figure, plt.Axes]:
    # 3D network plot
    plt.style.use('paper_style')
    #fig, ax = plt.subplots(constrained_layout=True) #UserWarning: This figure was using constrained_layout, but that is incompatible with subplots_adjust and/or tight_layout; disabling constrained_layout. fig.tight_layout()
    fig, ax = plt.subplots()
    fig = size_fig(fig, size_width, size_height)
    return fig, ax

def get_plt_style(size_width=3.25, size_height=3.25) ->plt:
    # 3D network plot
    plt.style.use('paper_style')
    # fig, ax = plt.subplots(constrained_layout=True)
    # fig = size_fig(fig, size_width, size_height)
    return plt

def keep_grid_and_remove_ticks_and_labels(fig, ax):
    ax.title.set_visible(False)
    ax.axis(True)
    ax.grid(True)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    ax.set_xlabel('')

    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    ax.set_ylabel('')
    return fig, ax

class GifMaker():
    def __init__(self, name_of_gif, duration = 0.4):
        if SAVE_RESULTS_BOOLEAN:
            filename_output = GIF_DIR+name_of_gif+".gif"
            self.writer = imageio.get_writer(filename_output, mode='I', duration=duration)
        else:
            print('If a gif is supposed to be created,\\'
                  ' set the variable PLOT_RESULTS_BOOLEAN in config_plots.py!')

    def add_figure(self, fig, ax=None):
        if SAVE_RESULTS_BOOLEAN:
            if AXIS_OFF and ax is not None:
                ax.axis('off')
            # dpi_resolution =100
            # fig.set_figheight(100)
            # fig.set_figwidth(100)
            # fig.set_dpi(dpi_resolution)
            canvas = FigureCanvasAgg(fig)
            # Retrieve a view on the renderer buffer
            canvas.draw()
            buf = canvas.buffer_rgba()
            # convert to a NumPy array
            fig_image = np.asarray(buf)
            # Negative input values will be clipped.
            # Positive values are scaled between 0 and 255.
            fig_image_ubyte = img_as_ubyte(fig_image)
            self.writer.append_data(fig_image_ubyte)
            # followed: https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
    def close_writer(self):
        if SAVE_RESULTS_BOOLEAN:
            self.writer.close()
        else:
            print('SAVE_RESULTS_BOOLEAN has to be set in config_plot.py')






