import os
import torch
from typing import Optional

from typing import NamedTuple, Sequence, Union
# datastructures
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex, SoftPhongShader, TensorProperties
)
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from utils.utils_sampel_points import _sample_points_from_meshes, _phong_sample_points_from_mesh
from utils.utils_sampel_points import _flip_sort_of_column, _rotate, _flip
from torch import nn

# def project_3d_points_to_image_plane_without_distortion(proj_matrix, points_3d, convert_back_to_euclidean=True):
#     """Project 3D points to image plane not taking into account distortion
#     Args:
#         proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
#         points_3d numpy array or torch tensor of shape (N, 3): 3D points
#         convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
#                                         NOTE: division by zero can be here if z = 0
#     Returns:
#         numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
#     """
#     if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
#         result = euclidean_to_homogeneous(points_3d) @ proj_matrix.T
#         if convert_back_to_euclidean:
#             result = homogeneous_to_euclidean(result)
#         return result
#     elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
#         result = euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
#         if convert_back_to_euclidean:
#             result = homogeneous_to_euclidean(result)
#         return result
#     else:
#         raise TypeError("Works only with numpy arrays and PyTorch tensors.")
