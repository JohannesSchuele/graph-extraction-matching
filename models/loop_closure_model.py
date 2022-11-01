import torch
import torch.nn as nn
import numpy as np
# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate
# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
)

import copy
from models.pose_rodrigues_rot_fromula import Camera3DPoseRodriguesRotFormula


class LoopClosureModel(nn.Module):

    def __init__(self, world_map, inverse_renderer, meshes_world, weight_normal_loss =0.1, print_loss=False, device='cpu'):
        super().__init__()
        self.world_map = world_map
        self.seen_images = world_map.get_seen_images
        self.device = device
        self.inverse_renderer = inverse_renderer
        self.mesh = meshes_world
        self.list_of_projected_points = list()
        # https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size (somewhere in the half)
        # [[]]*10 using that, you will run into referencing errors
        # different object reference each time
        self.corresponding_points_in_world_point_cloud = [[] for _ in range(world_map.world_point_cloud_map.points_packed().detach().shape[0])]
        self.corresponding_normals_in_world_point_cloud = [[] for _ in range(world_map.world_point_cloud_map.points_packed().detach().shape[0])]
        self.weight_normal_loss = weight_normal_loss

        self.print_loss = print_loss
        self.dist_init = None
        self.parameters_seen = None
        self.parameters_init = None
        self.param_r_rotation_batch = None
        self.param_t_translation_batch = None
        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=self.device)
        self.__define_parameters()
        self.DOFs_per_view = 2 * ((self.param_t_translation_batch.shape[0] - 1) * self.param_t_translation_batch.shape[1]) + 1

    def __define_parameters(self):
        # initialize the absolute log-rotations/translations with random-zero entries
        r_rotation_batch = torch.zeros((self.seen_images.__len__(), 3), dtype=torch.float32, device=self.device)
        t_translation_batch = torch.zeros((self.seen_images.__len__(), 3), dtype=torch.float32, device=self.device)
        for i, graph_images in enumerate(self.seen_images):
            graph_image = graph_images[2]
            r_rotation_batch[i, :] = graph_image.r_rotation
            t_translation_batch[i, :] = graph_image.t_translation
            if i == 0:
                # furthermore, we know that the first camera is a trivial one
                self.r_rotation_init = graph_image.r_rotation.clone().detach()
                self.t_translation_init = graph_image.t_translation.clone().detach()

        # the mask the specifies which cameras are going to be optimized
        #     (since we know the first camera is already correct,
        #      we only optimize over the 2nd-to-last cameras)
        self.camera_mask_init = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self.camera_mask_init[: ,2] = 0 #ToDo check if this needs to be zero or one, depending if an acutal loop closure is needed!
        self.param_r_rotation_batch = nn.Parameter(r_rotation_batch)
        self.param_t_translation_batch = nn.Parameter(t_translation_batch)



    def forward(self):
        ##
        # Deepcopy is needed in the following, else it would fail without retain_graph=True in optimization
        corresponding_points_in_world_point_cloud = copy.deepcopy(self.corresponding_points_in_world_point_cloud)
        corresponding_normals_in_world_point_cloud = copy.deepcopy(self.corresponding_normals_in_world_point_cloud)
        corresponding_points_in_world_point_cloud, corresponding_normals_in_world_point_cloud, cameras = self.re_project_all_node_positions(corresponding_points_in_world_point_cloud, corresponding_normals_in_world_point_cloud)
        euclidean_loss, normal_loss = self.calculate_loss(corresponding_points_in_world_point_cloud, corresponding_normals_in_world_point_cloud)
        loss = euclidean_loss+normal_loss
        if self.print_loss:
            print('Loss function : ', loss.cpu().detach().numpy(),
                  'Loss function distance : ', euclidean_loss.cpu().detach().numpy(),
                  'Loss function normal : ',  normal_loss.cpu().detach().numpy())
        return loss, cameras

    def re_project_all_node_positions(self, corresponding_points_in_world_point_cloud, corresponding_normals_in_world_point_cloud):
        R_batch = self.cam_rodrigues_object.get_rotation_matrix(self.param_r_rotation_batch)
        T_batch = self.cam_rodrigues_object.get_translation_matrix(self.param_t_translation_batch)
        for i, graph_images in enumerate(self.seen_images):
            graph_image = graph_images[2]
            print('iteration', i)
            if i == 0:
                R = self.cam_rodrigues_object.get_rotation_matrix(self.r_rotation_init)[0]
                T = (T_batch[0] * self.camera_mask_init + self.t_translation_init * (torch.ones((1, 3), dtype=torch.float32, device=self.device)-self.camera_mask_init))[0]
            else:
                R = R_batch[i]
                T = T_batch[i]

            #ToDo: Implement a mask here!! Values could easily get None!!!
            reprojected_point_cloud2, _ = self.inverse_renderer(meshes_world=self.mesh.clone(),
                                                                node_pos=graph_image.node_pos, R=R[None], T=T[None],
                                                                req_gradient=True, req_fragments=False)
            mask_points = torch.logical_not(torch.isnan(reprojected_point_cloud2.points_packed()))[:, 0]
            mask_normals = torch.logical_not(torch.isnan(reprojected_point_cloud2.normals_packed()))[:, 0]
            # two masks are needed! mask_normals and mask_points are actually not identical necessarily!
            mask = torch.all(torch.vstack((mask_points, mask_normals)), dim=0)

            if isinstance(graph_image.get_matches, np.ndarray):
                valid_match2 = copy.deepcopy(graph_image.get_matches.astype(int))
            elif torch.is_tensor(graph_image.get_matches):
                valid_match2 = copy.deepcopy(graph_image.get_matches.long())
            else:
                raise TypeError("Match-Arrays only with numpy arrays and PyTorch tensors.")
            corresponding_points_in_world_point_cloud, corresponding_normals_in_world_point_cloud = self._set_points(corresponding_points_in_world_point_cloud, corresponding_normals_in_world_point_cloud,
                                                                                                                   valid_match2, mask, reprojected_point_cloud2)
        with torch.no_grad():
            focal_length = self.inverse_renderer.rasterizer.cameras.focal_length.clone()
            cameras = self.inverse_renderer.rasterizer.cameras.__class__(focal_length=focal_length, R=R_batch, T=T_batch, device=self.device)

        return corresponding_points_in_world_point_cloud, corresponding_normals_in_world_point_cloud, cameras


    def _set_points(self, corresponding_points_in_world_point_cloud, corresponding_normals_in_world_point_cloud, matches, mask, reprojected_point_cloud):
        for i, match in enumerate(matches):
            if mask[match[1]]:
                corresponding_points_in_world_point_cloud[match[0]].append(reprojected_point_cloud.points_packed()[match[1]])
                corresponding_normals_in_world_point_cloud[match[0]].append(reprojected_point_cloud.normals_packed()[match[1]])

        return corresponding_points_in_world_point_cloud, corresponding_normals_in_world_point_cloud


    def calculate_loss(self, corresponding_points_in_world_point_cloud, corresponding_normals_in_world_point_cloud):
        # ToDo: save results of means in last iteration to update pointcloud positions and normals!0
        # euclidean_loss_vec = torch.tensor(0,device=self.device)
        # normal_loss =  torch.tensor(0,device=self.device)
        euclidean_loss_vec = torch.zeros(len(corresponding_points_in_world_point_cloud), device=self.device)
        normal_loss_vec = torch.zeros(len(corresponding_normals_in_world_point_cloud), device=self.device)

        for i, projected_points in enumerate(corresponding_points_in_world_point_cloud):
            if projected_points and len(projected_points) > 1:  # ensure that sublist is not empty!
                # size should be >1, else error would not make sense to calculate!
                # calculate euclidean loss
                projected_point_tensor = torch.stack(projected_points)
                center_of_gravity = torch.mean(projected_point_tensor, dim=0, keepdim=True)
                distance = (projected_point_tensor - center_of_gravity) ** 2
                #euclidean_loss_vec[i] = torch.sqrt(distance.mean())
                euclidean_loss_vec[i] = distance.mean()
                # ToDo compare loss as sum and loss as the mean of the vector

                # calculate normal loss
                projected_normal_tensor = torch.stack(corresponding_normals_in_world_point_cloud[i])
                #if torch.all(torch.logical_not(torch.isnan(mean_normal)))#torch.any((torch.isnan(mean_normal)))
                mean_normal = torch.mean(projected_normal_tensor, dim=0, keepdim=True)
                normal_loss_vec = 1 - (torch.abs(torch.cosine_similarity(projected_normal_tensor,
                                                                         mean_normal,
                                                                         dim=1))).mean()

        euclidean_loss = euclidean_loss_vec.mean()
        normal_loss = normal_loss_vec.mean()
        return euclidean_loss, normal_loss


    def update_world_map_with_latest_fragments_and_camera_positions(self):
        r_rotation_batch = self.param_r_rotation_batch.clone().detach()
        t_translation_batch = self.param_t_translation_batch.clone().detach()
        R_batch = self.cam_rodrigues_object.get_rotation_matrix(t_translation_batch)
        T_batch = self.cam_rodrigues_object.get_translation_matrix(self.param_t_translation_batch)
        for i, graph_images in enumerate(self.seen_images):
            graph_image = graph_images[2]
            print('iteration', i)
            if i == 0:
                R = self.cam_rodrigues_object.get_rotation_matrix(self.r_rotation_init)[0]
                T = (T_batch[0] * self.camera_mask_init + self.t_translation_init * (torch.ones((1, 3), dtype=torch.float32, device=self.device)-self.camera_mask_init))[0]
            else:
                R = R_batch[i]
                T = T_batch[i]

            #ToDo: check how to get a unique policy to update!!!
            assert False
            reprojected_point_cloud2, fragments_map = self.inverse_renderer(meshes_world=self.mesh.clone(),
                                                                node_pos=graph_image.node_pos, R=R[None], T=T[None],
                                                                req_gradient=True, req_fragments=True)




            mask_points = torch.logical_not(torch.isnan(reprojected_point_cloud2.points_packed()))[:, 0]
            mask_normals = torch.logical_not(torch.isnan(reprojected_point_cloud2.normals_packed()))[:, 0]
            # two masks are needed! mask_normals and mask_points are actually not identical necessarily!
            mask = torch.all(torch.vstack((mask_points, mask_normals)), dim=0)

            if isinstance(graph_image.get_matches, np.ndarray):
                valid_match2 = copy.deepcopy(graph_image.get_matches.astype(int))
            elif torch.is_tensor(graph_image.get_matches):
                valid_match2 = copy.deepcopy(graph_image.get_matches.long())
            else:
                raise TypeError("Match-Arrays only with numpy arrays and PyTorch tensors.")
            corresponding_points_in_world_point_cloud, corresponding_normals_in_world_point_cloud = self._set_points(corresponding_points_in_world_point_cloud, corresponding_normals_in_world_point_cloud,
                                                                                                                   valid_match2, mask, reprojected_point_cloud2)

        with torch.no_grad():
            focal_length = self.inverse_renderer.rasterizer.cameras.focal_length.clone()
            cameras = self.inverse_renderer.rasterizer.cameras.__class__(focal_length=focal_length, R=R_batch, T=T_batch, device=self.device)

        return world_map