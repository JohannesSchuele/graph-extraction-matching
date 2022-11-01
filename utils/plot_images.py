# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from pytorch3d.vis.plotly_vis import get_camera_wireframe
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    FoVPerspectiveCameras
)

from configs.plot.colours import *
from configs.plot.config_plots import *
import torch

# def visualize_image(img, size_width=3.75, size_height=3.75, title=''):
#     fig, ax = generate_mpl_figure(size_width=size_width, size_height=size_height)
#     ax.imshow(img)
#     ax.axis('on')
#     ax.set_title(title)
#     fig.tight_layout()
#     # fig.show()
#     return fig, ax

def visualize_image(img, size_width=3.75, size_height=3.75, title = ''):
    if torch.is_tensor(img):
        img = img[0].clone().cpu().detach().numpy()
    fig, ax = generate_mpl_figure(size_width=size_width, size_height=size_height)
    ax.imshow(img)
    ax.axis('on')
    ax.set_title(title)
    fig.tight_layout()
    fig.show()
    return fig, ax

def visualize_difference_of_grey_scale_images_by_rgb(img_original, img_desired, size_width=3.75, size_height=3.75, title=''):
    fig, ax = generate_mpl_figure(size_width=size_width, size_height=size_height)
    nz = np.zeros((img_original.shape[0], img_original.shape[0]))
    img_original_rgb = np.stack((img_original, img_original, img_original), axis=2)
    img_desired_rgb = np.stack((nz, img_desired, nz), axis=2)

    interscetion_img = img_original_rgb + img_desired_rgb
    fig, ax = generate_mpl_figure(size_width=3.5, size_height=3.5)
    # ax.imshow(img, cmap = 'gray', vmin = 0, vmax = 1.0)
    ax.imshow(interscetion_img)
    ax.axis('on')
    fig.tight_layout()

    return fig, ax

