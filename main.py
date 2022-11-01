import torch
from pytorch3d.utils import ico_sphere
import numpy as np
# Util function for loading meshes
# Data structures and functions for rendering
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer
)
import sys
import os
sys.path.append(os.path.abspath(''))
# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
import datetime
from tools_generate import Config, DataGenerator
time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
from tools_graph.utilz_analysis import get_data, plot_graph_on_img
from models.pose_rodrigues_rot_fromula import Camera3DPoseRodriguesRotFormula
from models.renderer import MeshRendererWithFragments2PointCloud
from models.renderer import SoftMesh2Image2PointCloudShader
from pytorch3d.renderer import BlendParams
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.loss import (
    mesh_laplacian_smoothing,
)

## Data Generator
conf = Config("config.yaml")
# generate data
training_generator = DataGenerator(conf, ImageType='masked')
image_rgb, image_grey, graph, pos, _, _, adj_matrix, _ = get_data(training_generator, batch_nr=0, item_in_batch=0)
fig1 = plot_graph_on_img(image_rgb, pos, adj_matrix)
fig2 = plot_graph_on_img(image_grey, pos, adj_matrix)


