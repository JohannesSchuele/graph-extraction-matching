import torch
from tqdm.notebook import tqdm
import torch.nn as nn

# io utils

# datastructures

# 3D transformations functions
# rendering components
from configs.plot.config_plots import *
from models.camera_pose_opt_model import PoseModel
from torch.cuda.amp import GradScaler


class CameraPoseOpt(nn.Module):
    """
    Reconstruct the camera pose by using the renderer "MeshRendererWithFragments2PointCloud"
    where the error loss is calculated based on the 3D located positions.
        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
    have one or more barycentric coordinates lying outside the range [0, 1].
    For a pixel with out of bounds barycentric coordinates with respect to a
    face f, clipping is required before interpolating the texture uv
    coordinates and z buffer so that the colors and depths are limited to
    the range for the corresponding face.
    For this set rasterizer.raster_settings.clip_barycentric_coords=True
    """
    def __init__(self, inverse_renderer, losses):
        super().__init__()
        if losses:
            self.losses = losses
        else:
            self.losses = {"euclidean": {"weight": 1.0, "values": []},
          "normal": {"weight": 800, "values": []},
         "pull-back": {"weight": 100, "values": []}
          }
        self.inverse_renderer = inverse_renderer
        self._ith_call=0

    def _set_optimizer(self, model, lr = 0.1):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        parameters_trainable, parameters_all = self.count_parameters(model)
        print('Camera model, contains ', parameters_trainable, 'trainable parameters and in total ', parameters_all, 'parameters!')
        return optimizer

    def _set_pose_opt_problem(self, meshes, matches, current_image_node_pos,
                              world_point_cloud_map, r_rotation_init, t_translation_init):
        camera_pose_model = PoseModel(meshes=meshes, inverse_renderer=self.inverse_renderer,
                                             match=matches,
                                             current_image_node_pos=current_image_node_pos,
                                             world_point_cloud_map=world_point_cloud_map,
                                             r_param_rotation=r_rotation_init,
                                             t_param_translation=t_translation_init,
                                             losses= self.losses,
                                             ith_call=self._ith_call
                                             )
        return camera_pose_model


    def camera_pose_forward(self, matches, meshes , world_point_cloud_map,current_image_node_pos,
                            r_rotation_init, t_translation_init, iterations=20, lr=0.01):
        self._ith_call += 1
        pose_opt_model = self._set_pose_opt_problem(meshes=meshes, matches=matches,
                                                    world_point_cloud_map=world_point_cloud_map,
                                                    current_image_node_pos=current_image_node_pos,
                                                    r_rotation_init=r_rotation_init,
                                                    t_translation_init=t_translation_init,
                                                    )
        optimizer = self._set_optimizer(model=pose_opt_model, lr=lr)

        # optimization loop:
        # loop = tqdm(range(iterations))
        # for i in loop:
        #     # Initialize optimizer
        #     optimizer.zero_grad()
        #     plot_bool = i % self.plot_period == 0
        #     # Runs the forward pass with autocasting.
        #     #with autocast():
        #     sum_loss = pose_opt_model.forward(plot_bool=plot_bool, iteration=i)
        #     print('loss :', sum_loss)
        #     # Print the losses
        #     loop.set_description("total_loss = %.6f" % sum_loss)
        #
        #     #sum_loss = 1000*torch.trunc(sum_loss*LOSS_PRECISION_DECIMAL_DIGITS)/LOSS_PRECISION_DECIMAL_DIGITS
        #     sum_loss.backward()
        #     optimizer.step()

        # Scaling used as in: https://pytorch.org/docs/stable/notes/amp_examples.html
        scaler = GradScaler()
        # optimization loop:
        loop = tqdm(range(iterations))
        for i in loop:
            # Initialize optimizer
            optimizer.zero_grad()
            # Runs the forward pass with autocasting.
            #with autocast():
            sum_loss = pose_opt_model.forward()
            # Print the losses
            loop.set_description("total_loss = %.6f" % sum_loss)

            # plot error between stitched and target point cloud on the mesh!
            if i % PLOT_PERIOD_POSE_OPT == 0 and PLOT_CAMERA_LOSS_ON_MESH:
                fig, _ = pose_opt_model.plot_pose_error_on_mesh_scene(ith_call=self._ith_call, ith_iteration=i)

            #sum_loss = 1000*torch.trunc(sum_loss*LOSS_PRECISION_DECIMAL_DIGITS)/LOSS_PRECISION_DECIMAL_DIGITS
            scaler.scale(sum_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #optimizer.step()l

        r_param_rotation_opt, t_param_translation_opt = pose_opt_model.get_parameters_rotation_and_translation
        r_param_rotation_opt.requires_grad = False
        t_param_translation_opt.requires_grad = False

        if PLOT_LOSS_CAMERA_POSE:
            fig_plot_losses = plot_losses(losses=pose_opt_model.get_loss_dict_history)
            fig_plot_losses = plot_losses(losses=pose_opt_model.get_loss_dict_history)
            save_figure(fig=fig_plot_losses, name_of_figure='camera_pose_opt_losses_'+str(self._ith_call)+'_ith_call')
        if MAKE_GIF_CAMERA_POSE_ERROR:
            pose_opt_model.close_gif_maker()
            #pose_opt_model.close_gif_maker()

        return r_param_rotation_opt.type(torch.float), t_param_translation_opt.type(torch.float)

    @staticmethod
    def count_parameters(model):
        parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        parameters_all = sum(p.numel() for p in model.parameters())
        return parameters_trainable, parameters_all


def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l['values'], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")
    plt.show()
    return fig



