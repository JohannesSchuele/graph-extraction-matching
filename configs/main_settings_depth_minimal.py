
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from pytorch3d.utils import ico_sphere
# Util function for loading meshes
# Data structures and functions for rendering
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer
)
import sys

sys.path.append(os.path.abspath(''))
# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

print('Number of available cuda devices: ', torch.cuda.device_count())
#device = torch.device("cpu")


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
from models.deformation import Deformation
from utils.mesh_operations import subdivideMesh, pool_mesh_to_dim_of_original_mesh
from tools_generate.DataDepthMap import *





## Data Generator
def adapt_relative_path_to_path_of_call(path):
    if os.getcwd().__len__() == 69: # to get it working from the examples folder in Tuebingen
        PATH = '../'+path.replace('./', '')
    elif os.getcwd().__len__() == 39:  # to get it working from the examples folder in Stuttgart
        PATH = '../' + path.replace('./', '')
    else: # by default from the main project level
        PATH = path

    return PATH

# PATH = adapt_relative_path_to_path_of_call(path="config.yaml")
# conf = Config(PATH)
# # generate data
# training_generator = DataGenerator(conf, ImageType='masked')
# image_rgb, image_grey, graph, pos, _, _, adj_matrix, _ = get_data(training_generator, batch_nr=0, item_in_batch=0)
# fig1 = plot_graph_on_img(image_rgb, pos, adj_matrix)
# fig2 = plot_graph_on_img(image_grey, pos, adj_matrix)

# Set paths
DATA_DIR= adapt_relative_path_to_path_of_call(path="./data")
# DATA_DIR = "./data"

# set image_size
image_size = 128

#sigma = 1e-4
##sigma = 0.0
## sigma s a positive scalar that controls the sharpness of the probability distribution
## sigma -> 0 the shaper the distribution gets
#raster_settings_soft = RasterizationSettings(
#    image_size= image_size,
#    #blur_radius=np.log(1. / 1e-4 - 1.)*1*sigma*10,
#    bin_size=20,
#    blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
#    faces_per_pixel=50,#ToDo test effect also with different focal_lengths
#    perspective_correct = True,#ToDo check all that
#    clip_barycentric_coords= False, # ToDo check the effect of this rasterizing setting if it is actually needed here!
#    cull_backfaces = True,
#)
#
#raster_settings_soft = RasterizationSettings(
#    image_size= image_size,
#    #blur_radius=np.log(1. / 1e-4 - 1.)*1*sigma*10,
#    bin_size=40,
#    blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
#    faces_per_pixel=50,#ToDo test effect also with different focal_lengths
#    perspective_correct = True,#ToDo check all that
#    clip_barycentric_coords= False, # ToDo check the effect of this rasterizing setting if it is actually needed here!
#    cull_backfaces = True,
#)
#
## raster_settings_soft = RasterizationSettings(
##     image_size= image_size,
##     #blur_radius=np.log(1. / 1e-4 - 1.)*1*sigma*10,
##     bin_size=10,
##     blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
##     faces_per_pixel=1,#ToDo test effect also with different focal_lengths
##     perspective_correct=False,#ToDo check all that
##     clip_barycentric_coords=False, # ToDo check the effect of this rasterizing setting if it is actually needed here!
##     cull_backfaces=False,
## )
#print('Renderer Settings: image_size: ',raster_settings_soft.image_size,'blur_radius: ', raster_settings_soft.blur_radius,'faces_per_pixel: ', raster_settings_soft.faces_per_pixel)
# Differentiable soft renderer using per vertex RGB colors for texture

#------------------------------
# R_cv = torch.from_numpy((np.eye(3)))[None].type(dtype=torch.float32)
# R_cv[:, 0, 0] = -1.0
# R_cv[:, 1, 1] = -1.0
# t_cv= torch.from_numpy((np.array([-2.0, -2.0, 2.0])))[None].type(dtype=torch.float32)
# camera_matrix_cv= torch.from_numpy((np.array([[330.0, 0.0, 332.0],[0.0, 330.0, 333.0], [0.0, 0.0, 1.0]])))[None].type(dtype=torch.float32)
# camera_matrix_cv= torch.from_numpy((np.array([[290.0, 0.0, 332.0],[0.0, 290.0, 333.0], [0.0, 0.0, 1.0]])))[None].type(dtype=torch.float32)
# #camera_matrix_cv= torch.from_numpy((np.array([[300.0, 0.0, 0.0],[0.0, 300.0, 0.0], [0.0, 0.0, 1.0]])))[None].type(dtype=torch.float32)
# image_size_cv = torch.tensor(([image_size, image_size]))[None].type(dtype=torch.float32)
# camera = cameras_from_opencv_projection(
#     R_cv,
#     t_cv,
#     camera_matrix_cv,
#     image_size_cv)
# camera.focal_length = camera.focal_length.clone()*(image_size/128)
# camera.principal_point = (-1.6, -1.6)
# #ToDo: fix that!! principle point shouldn't be zero by default!
# camera.principal_point = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
# camera = camera.float()
# camera = camera.to(device=device)
# cam_rot = Camera3DPoseRodriguesRotFormula(N=1, device=device)
# # initialize the absolute log-rotations/translations
# # initial view is defined in pytorch3ds coordinate system!!
# log_R_absolute= cam_rot.get_rot_vec_from_rotation_matrix(camera.R)
# log_R_absolute= cam_rot.get_rot_vec_from_rotation_matrix(camera.R)
# log_R_absolute[:, 0]= np.pi
# log_R_absolute[:, 1]= 1.1
# log_R_absolute[:, 2]= 0.8
# T_absolute = camera.T
#
#
# log_R_absolute[:, 0]= np.pi
# log_R_absolute[:, 1]= 0.0
# log_R_absolute[:, 2]= 0.0
# cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=device)
# R = cam_rodrigues_object.get_rotation_matrix(log_R_absolute)
# T_absolute = camera.T
#
# T_absolute[:, 0] = 0.0
# T_absolute[:, 1] = 0
# T_absolute[:, 2] = -5.0  # in camera view back off of object in negative direction
#
# T = torch.matmul(T_absolute, R.inverse())[0]
#
#
#
# N = 1
# camera_mask = torch.ones(N, 1, dtype=torch.float32, device=device)
# camera_mask[0] = 0.
# R_absolute = R
# T_absolute = T
# # get the current absolute cameras
# camera.R = R_absolute
# camera.T = T_absolute
#####
# R_cv, t_cv, K_cv = opencv_from_cameras_projection(cameras=target_cameras, image_size=torch.tensor([image_size, image_size])[None])
# ####
# inverse_renderer = MeshRendererWithFragments2PointCloud(
#     rasterizer=MeshRasterizer(
#         cameras=camera,
#         raster_settings=raster_settings_soft
#     ),
#     shader = SoftMesh2Image2PointCloudShader(
#         cameras=camera,
#         blend_params = BlendParams(sigma=sigma, gamma=4*1e-2)
#         # gamma->0, the aggregation function only outputs the color of the nearest triangle,
#         # which exactly matches the behavior of z-buffering.
#     )
# )

# ------------------------------------------------------------------------------
from models.texture import InitRenderers

renderer_settings = {"face per pixel - opt": 15,
                     "blur radius - opt": np.log(1. / 1e-4 - 1.)*1e-4,
                     "blend param sigma - opt": 1e-4,
                     "blend param gamma - opt": 5*1e-4,

                     "face per pixel - high opt": 50,
                     "blur radius - high opt": np.log(1. / 1e-4 - 1.)*1e-3,
                     "blend param sigma - high opt": 1e-4,
                     "blend param gamma - high opt": 5 * 1e-2,

                     "face per pixel - view": 10,
                     "blur radius - view": 1e-6,
                     "blend param sigma - view": None,
                     "blend param gamma - view": None,
                     "face per pixel - inverse": 15,
                     "blur radius - inverse": np.log(1. / 1e-4 - 1.) * 1e-4,
                     "blend param sigma - inverse": 1e-4,
                     "blend param gamma - inverse": 1e-5,
                     "image size view rendering": 256,

                     "face per pixel - inverse nodePoseOpt": 40,
                     "blur radius - inverse nodePoseOpt": np.log(1. / 1e-4 - 1.) * 1e-4,
                     "blend param sigma - inverse nodePoseOpt": 1e-4,
                     "blend param gamma - inverse nodePoseOpt": 4 * 1e-2,
                     }

# camera_init = camera
from pytorch3d.renderer.cameras import PerspectiveCameras
cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, device=device)
R = cam_rodrigues_object.get_rotation_matrix(torch.tensor([-3.1,0,0], device=device))
# R = cam_rodrigues_object.get_rot_vec_from_rotation_matrix(R_matrix=h)
# R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]],
#                  device=device, dtype=torch.float)[None],
camera_init = PerspectiveCameras(device=device,
                                 R= torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                                                 device=device, dtype=torch.float)[None],
                                 T=torch.tensor([0, 0, 2.5], device=device, dtype=torch.float)[None],
                                 principal_point=((0, 0),),
                                 focal_length=((4, 4),))

RenderersCollection = InitRenderers(camera_texture=camera_init, image_size=image_size, camera_view=None,
                                    renderer_settings=renderer_settings, device=device)
