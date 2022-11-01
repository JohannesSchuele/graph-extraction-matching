import torch
import torch.nn as nn
import networkx as nx
from models.pose_rodrigues_rot_fromula import Camera3DPoseRodriguesRotFormula
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.renderer import (
    PointLights,
    AmbientLights
)
from pytorch3d.renderer.cameras import (
    PerspectiveCameras,
)
from models.multiview import CameraTorch, calc_reprojection_error_matrix, \
    project_3d_points_to_image_plane_without_distortion
from pytorch3d.utils import opencv_from_cameras_projection
from tools_graph.utilz_analysis import load_data_sample
from tools_generate.node_classifiers import NodeTypes
from models.point_cloud_features import PointCloudMap
from configs.plot.config_plots import *
from configs.plot.colours import *
from mpl_toolkits.mplot3d import Axes3D
from utils.mesh_operations import pool_mesh_to_dim_of_original_mesh
from models.pc_alignment_pose_opt import scale_point_cloud
import cv2
from configs.config import *
import math

class NodePointContainer(nn.Module):

    def __init__(self, world_point_cloud_map, matches_with_world_map_list,
                 points_2D_map_list, split_indices_crossing_list=None, split_indices_border_list=None):

        super().__init__()
        self.device= world_point_cloud_map.device
        self.world_point_cloud_map = world_point_cloud_map
        self.matches_with_world_map_list = matches_with_world_map_list
        self.points_2D_map_list = [torch.tensor(points_2D_map_list[i]).to(device=self.device) for i in range(points_2D_map_list.__len__())]
        self.split_indices_crossing_list = split_indices_crossing_list
        self.split_indices_border_list = split_indices_border_list

    def get_updated_node_points_based_on_mesh(self, geometry_mesh):

        updated_point_cloud = self.world_point_cloud_map.update_point_cloud_based_on_mesh(mesh=geometry_mesh)
        return updated_point_cloud.points_packed()

    @property
    def get_node_points_on_mesh(self):
        return self.world_point_cloud_map.points_packed()

    @property
    def get_matches_with_world_map_list(self):
        return self.matches_with_world_map_list

    @property
    def get_points_2D_map_list(self):
        return self.points_2D_map_list

    @property
    def get_split_indices_crossing_list(self):
        return self.split_indices_crossing_list

    @property
    def get_split_indices_border_list(self):
        return self.split_indices_border_list

    def does_node_container_handle_splits(self):
        return self.split_indices_border_list is not None and self.split_indices_crossing_list is not None


def get_point_cloud(img_torch, mask_torch=None, threshold=0.4, idx_matrix=None):
    if idx_matrix is None:
        idx_matrix = get_index_matrix(device=img_torch.device)

    if mask_torch is None:
        mask = torch.gt(img_torch[0, :, :, 0], threshold)
    else:
        mask = torch.bitwise_and(torch.gt(img_torch[0, :, :, 0], threshold), mask_torch[0])
    idx_matrix = idx_matrix[mask]
    # idx_matrix = ((img_torch[0, :, :, 0] > 0.2) & (mask_torch[0] == True)).nonzero().squeeze()
    return idx_matrix


def get_point_clouds_of_selected_pairs(device):
    """ For testing: get point cloud of selected points on image
        Returns: x (predicted) and y (point clouds)"""
    choice = 'random'
    if choice == "random":
        torch.manual_seed(0)
        x = torch.rand((25, 2), device=device) * 113 + 5 # min Abstand 5 zum Rand (128)
        y_test = []
        for i, point in enumerate(x):
            y_test += [[point[0] - 0.1 * (point[0]-64), point[1] - 0.1 * (point[1]-64)]]
        y = torch.tensor(y_test, device=device)

    elif choice == 'grid':
        x_test, y_test = [], []
        for k in range(10, 118, 20):
            for l in range(10, 118, 20):
                x_test += [[k, l]]
                y_test += [[k + 0.1 * (k-64), l + 0.1 * (l-64)]]

        x = torch.tensor(x_test, device=device)
        y = torch.tensor(y_test, device=device)

    x = x.to(dtype=torch.int, device=device)
    y = y.to(dtype=torch.int, device=device)
    # numpy_img = img_torch.detach().cpu().numpy()[0, :, :]
    return x, y


def rotateImage(image, angle):
    row, col = image.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def get_index_matrix(device='cpu', image_size_int:int=128):
    """
    Returns: 3D Matrix - 2D with [row, col] ids in corresponding position ?
    """
    if type(image_size_int) is tuple and len(image_size_int) == 2:
        idx_matrix = torch.zeros((*image_size_int, 2))
        for i in range(idx_matrix.shape[0]):
            idx_matrix[i, :, 0] = torch.full((1, image_size_int[1]), i)[0]
            idx_matrix[i, :, 1] = torch.arange(0, image_size_int[1])
    else:
        idx_matrix = torch.zeros((image_size_int, image_size_int, 2))
        for i in range(idx_matrix.shape[0]):
            idx_matrix[i, :, 0] = torch.full((1, image_size_int), i)[0]
            idx_matrix[i, :, 1] = torch.arange(0, image_size_int)

    return idx_matrix.to(device=device)