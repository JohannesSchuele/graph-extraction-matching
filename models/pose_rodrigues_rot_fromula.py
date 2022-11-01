import torch
from pytorch3d.transforms.so3 import (
    so3_exp_map,
    so3_relative_angle,  so3_log_map
)
from pytorch3d.renderer.cameras import (
    PerspectiveCameras,
)

# add path for demo utils
import sys
import os
sys.path.append(os.path.abspath(''))
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")
from torch import nn


# # initialize the absolute log-rotations/translations with random entries
# log_R_absolute_init = torch.randn(N, 3, dtype=torch.float32, device=device)
# T_absolute_init = torch.randn(N, 3, dtype=torch.float32, device=device)
#
# # furthermore, we know that the first camera is a trivial one
# #    (see the description above)
# log_R_absolute_init[0, :] = 0.
# T_absolute_init[0, :] = 0.


class Camera3DPoseRodriguesRotFormula(nn.Module):
    # Using Rodrigues rotation formula
    # https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html
    # https://iopscience.iop.org/chapter/978-0-7503-1454-1/bk978-0-7503-1454-1ch5.pdf
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    def __init__(self, N, with_cam_mask = False, device='cpu'):
        super().__init__()
        self.device = device
        self.with_cam_mask = with_cam_mask
        if self.with_cam_mask:
            camera_mask = torch.ones(N, 1, dtype=torch.float32, device=device)
            camera_mask[0] = 0.
        else:
            camera_mask = torch.ones(N, 1, dtype=torch.float32, device=device)
        self.camera_mask = camera_mask

    def get_rotation_matrix(self, log_R_absolute: torch.Tensor):
        R_absolute = so3_exp_map(log_R_absolute * self.camera_mask)
        return R_absolute

    def get_rot_vec_from_rotation_matrix(self, R_matrix):
        log_rot = so3_log_map(R=R_matrix, eps=1e-15, cos_bound=0.0) #to prevent some transformation errors eps had to get increased!
        return log_rot

    def get_translation_matrix(self, T_absolute: torch.Tensor):
        T_absolute = T_absolute * self.camera_mask
        return T_absolute
    def get_rotation_and_translation(self, log_R_absolute, T_absolute):
        T_absolute = T_absolute * self.camera_mask
        R_absolute = so3_exp_map(log_R_absolute * self.camera_mask) #to prevent some transformation errors eps had to get increased!
        return R_absolute, T_absolute

    def get_camera_matrix(self, log_R_absolute, T_absolute):
        R_absolute = so3_exp_map(log_R_absolute * self.camera_mask)
        # get the current absolute cameras
        cameras_absolute = PerspectiveCameras(
            R=self.get_rotation_matrix(R_absolute),
            T=T_absolute * self.camera_mask, device=self.device)

        return cameras_absolute