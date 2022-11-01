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
from pytorch3d.renderer.cameras import (
    PerspectiveCameras,
)
from pytorch3d.renderer import (
    PointLights,
    AmbientLights
)
from configs.plot.config_plots import *
class CameraPoseModel_RGB(nn.Module):

    def __init__(self, seen_images, renderer,  mesh_model_vertex_rgb, losses, last_images_2bused: int = None, device='cpu'):
        super().__init__()

        self.seen_images = seen_images
        self.device = device
        self.image_renderer = renderer
        self.focal_length = renderer.rasterizer.cameras.focal_length
        self.mesh =  mesh_model_vertex_rgb
        self.list_of_projected_points = list()
        if losses is not None:
            self.losses = losses
        else:
            self.losses = {
                "rgb": {"weight": 0.5, "values": []},
                }

        self.last_images_2bused = last_images_2bused
        self.parameters_seen = None
        self.param_r_rotation_batch = None
        self.param_t_translation_batch = None
        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=self.device)
        self.__define_parameters()

    def __define_parameters(self):
        # initialize the absolute log-rotations/translations with random-zero entries
        if self.last_images_2bused is None:
            self.last_images_applied = self.seen_images.__len__()
        else:
            self.last_images_applied = min(self.last_images_2bused, self.seen_images.__len__())

        r_rotation_batch = torch.zeros((self.last_images_applied, 3), dtype=torch.float32, device=self.device)
        t_translation_batch = torch.zeros((self.last_images_applied, 3), dtype=torch.float32, device=self.device)
        # initial pose parameters from previous optimizations!
        for i, graph_images in enumerate(self.seen_images[: self.last_images_applied]):
            graph_image = graph_images[2]
            r_rotation_batch[i, :] = graph_image.r_rotation
            t_translation_batch[i, :] = graph_image.t_translation

        self.param_r_rotation_batch = nn.Parameter(r_rotation_batch)
        self.param_t_translation_batch = nn.Parameter(t_translation_batch)



    def forward(self, iteration):
        loss = {k: torch.tensor(0.0, device=self.device) for k in self.losses}
        loss["rgb"] = self.get_rendered_image_loss(mesh=self.mesh, iteration=iteration)

        sum_loss = torch.tensor(0.0, device=self.device)
        for k, l in loss.items():
            weighted_loss = l * self.losses[k]["weight"]
            sum_loss += weighted_loss
            self.losses[k]["values"].append(float(weighted_loss.detach().cpu()))

        return sum_loss

    @ property
    def get_last_images_applied(self):
        return self.last_images_applied


    def get_rendered_image_loss(self, mesh, iteration):
        R_batch = self.cam_rodrigues_object.get_rotation_matrix(self.param_r_rotation_batch)
        T_batch = self.cam_rodrigues_object.get_translation_matrix(self.param_t_translation_batch)

        cameras = PerspectiveCameras(device=self.device, R=R_batch,
                           T=T_batch, focal_length=self.focal_length)

        lights = PointLights(device=self.device, location=[[2.0, 2.0, -2.0]])  # ToDo: move that in a config file!
        ambient_light = AmbientLights(device=self.device)
        images_predicted = self.image_renderer(mesh, cameras=cameras, lights=lights)
        if PLOT_PREDICTED_TO_TARGET_IMAGE_DEFORMATION and iteration % PLOT_PREDICTED_TO_TARGET_IMAGE_POSE_OPTIMIZATION_PERIOD == 0:
            visualize_image_pose_opt_with_target(predicted_images=images_predicted,
                                                    target_image=self.target_image_batch,
                                                    title='Predicted to target image for pose optimization - at iteration: ' + str(
                                                        iteration),
                                                    silhouette=False)
        rgb_loss = ((images_predicted[..., :3] - self.target_image_batch) ** 2).mean()
        return rgb_loss

    @ property
    def get_rotation_translation_parameters(self):
        return self.param_r_rotation_batch, self.param_t_translation_batch



def visualize_image_pose_opt_with_target(predicted_images,
                         target_image, title='',
                         silhouette=False):
    inds = 3 if silhouette else range(3)
    item_in_batch = 0
    #with torch.no_grad():
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[item_in_batch, ..., inds].cpu().detach().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(target_image[item_in_batch].cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()