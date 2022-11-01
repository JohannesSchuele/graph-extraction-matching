
import sys
import os
sys.path.append(os.path.abspath(''))
# Setup
import datetime
from tools_generate import Config, DataGenerator
time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
from tools_graph.utilz_analysis import get_data, plot_graph_on_img
from tools_generate.distortion import *
from models.world_map_matching import WorldMapMatching
import numpy as np
import torch
import torch.nn as nn
# rendering components
from pytorch3d.renderer import (
    PerspectiveCameras
)
from models.deformation import Deformation
import copy
from models.point_cloud_features import PointCloudMap
from tools_graph.match_orb_tools import match
from tools_graph.plot_functs import plot_graph_matches_color
from models.ransac import triangulate_ransac, calibrate_intrinsic_camera_parameters
from models.pose_rodrigues_rot_fromula import Camera3DPoseRodriguesRotFormula
import networkx as nx
from utils.camera_visualization import plot_cameras
from models.camera_pose import camera_projection
from tools_generate.node_classifiers import NodeTypes
from configs.plot.colours import *
from configs.plot.config_plots import *
from configs.config import *
from models.camera_pose_opt import CameraPoseOpt
from utils.world_map_visualization import create_world_map_mesh_and_graph_figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tools_generate.UpComingDataGenerator import UpComingDataGenerator
from utils.mesh_operations import subdivideMesh
from models.texture import TextureTypes
from models.texture import Texture
from tools_generate.distortion import distort_image
from utils.deformation_helpers import NodePointContainer
from tools_generate.image_processing import skeletonise_and_clean
from utils.deformation_helpers import rotateImage
from utils.plot_images import visualize_image, visualize_difference_of_grey_scale_images_by_rgb
from utils.world_map_visualization import plot_mesh
from models.texture import InitRenderers
## Data Generator
conf = Config("config.yaml")
# generate data
training_generator = DataGenerator(conf, ImageType='masked')
image_rgb, image_grey, graph, pos, _, _, adj_matrix, _ = get_data(training_generator, batch_nr=0, item_in_batch=0)
fig1 = plot_graph_on_img(image_rgb, pos, adj_matrix)
fig2 = plot_graph_on_img(image_grey, pos, adj_matrix)


from models.world_map import WorldMap
distortion_param = get_camera_params_distortion()
undist_pos, image_grey_distorted2 = distort_image(cameraparams=get_camera_params(), image=current_image_grey,
                                                  input_pos=current_graph_image.get_all_image_nodes, plot_it=True)

rotated_image = rotateImage(image_grey_distorted2, angle=-5)

# undist_pos, mask_for_image_distorted = distort_image(cameraparams=get_camera_params(),
#                                                      image=,
#                                                      input_pos=,
#                                                      plot_it=True)
#
# undist_pos, matching_mask_for_image_distorted = distort_image(cameraparams=get_camera_params(),
#                                                               image=self.current_graph_image.generate_matching_mask_for_image(),
#                                                               input_pos=self.current_graph_image.get_all_image_nodes,
#                                                               plot_it=True)
world_map_matching = WorldMapMatching(data_generator= training_generator,
                             calibrate_cameras= False)
