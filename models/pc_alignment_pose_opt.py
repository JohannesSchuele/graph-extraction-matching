import torch
from tqdm.notebook import tqdm
import torch.nn as nn

# io utils

# datastructures

# 3D transformations functions

# rendering components
from models.pose_rodrigues_rot_fromula import Camera3DPoseRodriguesRotFormula
from configs.plot.colours import *
from configs.plot.config_plots import *
from pytorch3d.renderer.cameras import PerspectiveCameras
from utils.camera_visualization import plot_cameras



class PointCloudAlignmentOptimizationModel(nn.Module):
    def __init__(self, point_cloud_with_desired_pose, point_cloud_2b_transformed, R_measurement_pc_init=None, T_measurement_pc_init=None, losses=None, device='cpu', overall_point_clouds=None, camera_init_plot=None):
        super().__init__()
        self.point_cloud_with_desired_pose = point_cloud_with_desired_pose
        self.point_cloud_2b_transformed = point_cloud_2b_transformed
        self.device=device

        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, device=device)

        if R_measurement_pc_init is None or T_measurement_pc_init is None:
            r_init_pc = torch.zeros((1, 3), device=self.device)
            t_init_pc = torch.zeros((1, 3), device=self.device)
        else:
            r_init_pc = self.cam_rodrigues_object.get_rot_vec_from_rotation_matrix(R_matrix=R_measurement_pc_init)
            t_init_pc = self.cam_rodrigues_object.get_translation_matrix(T_absolute=T_measurement_pc_init)

        self.r_param_rotation = nn.Parameter(r_init_pc.to(self.device).type(torch.float))
        self.t_param_translation = nn.Parameter(t_init_pc.to(self.device).type(torch.float))

        if losses is None:
            self.losses = {"euclidean distance": {"weight": 1.0, "values": []},
                           "scale xy-plane ": {"weight": 1.0, "values": []}}
        else:
            self.losses = losses
        self.overall_point_clouds = overall_point_clouds
        self.opt_xy_scale = self.losses["scale xy-plane "]["weight"] > 0

        self.scale_xy_plane_param = torch.tensor(1.0, device=self.device, dtype=torch.float)
        if self.opt_xy_scale:
            self.scale_xy_plane_param = nn.Parameter(torch.tensor(1.0, device=self.device, dtype=torch.float))
        self.camera_init_plot = camera_init_plot


    def forward(self, plot_bool=False, iteration=0, mesh_for_plotting = None):
        R = self.cam_rodrigues_object.get_rotation_matrix(self.r_param_rotation)
        T = self.cam_rodrigues_object.get_translation_matrix(self.t_param_translation)

        if self.opt_xy_scale:
            point_cloud_2b_transformed = scale_point_cloud(torch_point_cloud=self.point_cloud_2b_transformed, scale_xy_plane=self.scale_xy_plane_param,
                                                               scale_z_direction=None)
        else:
            point_cloud_2b_transformed = self.point_cloud_2b_transformed

        point_cloud_transformed = torch.add(torch.matmul(point_cloud_2b_transformed, R), T)
        loss_vec = (self.point_cloud_with_desired_pose - point_cloud_transformed)**2

        ## get normal loss
        pd1 = self.point_cloud_with_desired_pose[0][0]
        pd2 = self.point_cloud_with_desired_pose[0][1]
        pd3 = self.point_cloud_with_desired_pose[0][2]

        pt1 = point_cloud_transformed[0][0]
        pt2 = point_cloud_transformed[0][1]
        pt3 = point_cloud_transformed[0][2]
        normal_desired = get_normal_from_spanned_triangle_torch(pd1, pd2, pd3)
        normal_target = get_normal_from_spanned_triangle_torch(pt1, pt2, pt3)
        loss_normal = 1 - torch.cosine_similarity(normal_desired[None], normal_target[None])

        loss_vec[:, 4, :] = loss_vec[:, 4, :] *0.01 # lets weight the central key point less!! - # ToDo parameter in config file!
        loss = loss_vec.mean() * self.losses['euclidean distance']["weight"] + 0.0* loss_normal
        self.losses['euclidean distance']["values"].append(float(loss.detach().cpu()))

        #ToDo: plot the point cloud

        if plot_bool:
            with torch.no_grad():
                point_cloud_measured_scaled = scale_point_cloud(torch_point_cloud=self.overall_point_clouds[1].clone().detach(), scale_xy_plane=self.scale_xy_plane_param,
                                                               scale_z_direction=None)
                point_cloud_measured_transformed = torch.add(torch.matmul(point_cloud_measured_scaled, R.clone().detach()), T.clone().detach()).detach()
                #for pc, pc_title in zip([self.overall_point_clouds[1].clone().detach(), point_cloud_measured_scaled,
                #                         point_cloud_measured_transformed], ['overall', 'scaled', 'transformed']):
                #    plot_torch_3dtensor(pc, title=pc_title)  # only to check the consistency of the pc

                fig = plt.figure(figsize=(10, 7))
                ax = Axes3D(fig)
                if mesh_for_plotting is not None:
                    from utils.world_map_visualization import plot_mesh
                    fig, ax = fig, ax = plot_mesh(mesh_model=mesh_for_plotting, fig=fig, ax=ax)
                fig, ax = plot_point_clouds_with_error(torch_point_cloud_current=point_cloud_transformed, torch_point_cloud_target_from_mesh=self.point_cloud_with_desired_pose,
                                             overall_point_clouds=[self.overall_point_clouds[0].detach(), point_cloud_measured_transformed], fig=fig, ax=ax)


                #camera = PerspectiveCameras(R=R.clone().detach(), T=T.clone().detach(), focal_length=2.0)
                #camera.to(device=self.device)
                if self.camera_init_plot is not None:
                    handle_cam = plot_cameras(ax=ax, cameras_input=self.camera_init_plot, color=blue_5)

                ax.view_init(elev=90, azim=0)
                #ax.view_init(elev=0, azim=0)

                fig.show()

                ax.view_init(elev=0, azim=90)
                fig.show()

        return loss





    @property
    def get_losses(self):
        return self.losses

    @property
    def get_parameters_rotation_and_translation(self):
        return self.r_param_rotation, self.t_param_translation

    @property
    def get_scale_xy_plane(self):
        return self.scale_xy_plane_param

def scale_point_cloud(torch_point_cloud, scale_xy_plane=None, scale_z_direction=None):
    _scale_matrix = torch.eye((3), dtype=torch.float, device=torch_point_cloud.device)
    if scale_xy_plane is not None:
        _scale_matrix[0, 0] = scale_xy_plane
        _scale_matrix[1, 1] = scale_xy_plane
    if scale_z_direction is not None:
        _scale_matrix[2, 2] = scale_z_direction

    return torch.matmul(torch_point_cloud, _scale_matrix)

class PointCloudAlignmentOptimization(nn.Module):
    def __init__(self, losses=None, device='cpu', plot_period=25, camera_view_model_synthetic = None):
        super().__init__()

        if losses is None:
            self.losses = {"euclidean distance": {"weight": 1.0, "values": []},
                           "scale xy-plane ": {"weight": 1.0, "values": []}}
        else:
            self.losses = losses

        self.device= device
        self.losses_results = {"euclidean distance": {"weight": 1.0, "values": []},
                                "scale xy-plane ": {"weight": 1.0, "values": []}}
        self._ith_call = 0
        self.scale_xy_plane = None
        self.camera_view_model_synthetic = camera_view_model_synthetic


    def _set_optimizer(self, model, lr=0.1):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        parameters_trainable, parameters_all = self.count_parameters(model)
        print('Camera model, contains ', parameters_trainable, 'trainable parameters and in total ', parameters_all,
              'parameters!')
        return optimizer

    def _set_alignment_opt_problem(self, point_cloud_with_desired_pose, point_cloud_keys_2b_aligned, R_measurement_pc_init=None, T_measurement_pc_init=None, losses=None, overall_point_clouds=None):
        pointcloud_alignment_model = PointCloudAlignmentOptimizationModel(point_cloud_with_desired_pose, point_cloud_keys_2b_aligned,
                                                                            R_measurement_pc_init=R_measurement_pc_init, T_measurement_pc_init=T_measurement_pc_init,
                                                                            losses=losses, device=self.device,
                                                                            overall_point_clouds=overall_point_clouds,
                                                                            camera_init_plot=self.camera_view_model_synthetic )
        return pointcloud_alignment_model

    def forward(self, point_cloud_with_desired_pose, point_cloud_keys_2b_aligned, R_measurement_pc_init=None, T_measurement_pc_init=None, iterations=20, lr=0.01, overall_point_clouds=None, mesh_for_plotting=None):
        self._ith_call += 1
        pointcloud_alignment_model = self._set_alignment_opt_problem(point_cloud_with_desired_pose, point_cloud_keys_2b_aligned,
                                                                     R_measurement_pc_init=R_measurement_pc_init, T_measurement_pc_init=T_measurement_pc_init,
                                                                     losses=self.losses, overall_point_clouds= overall_point_clouds)

        optimizer = self._set_optimizer(model=pointcloud_alignment_model, lr=lr)

        #optimization loop:
        loop = tqdm(range(iterations))
        for i in loop:
            # Initialize optimizer
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                plot_bool = i % PLOT_PERIOD_POINT_CLOUD_ALIGNMENT == 0 and PLOT_POINT_CLOUD_ALIGNMENT
                #plot_bool = True
                plot_bool = (i ==0) or (i == iterations-1)
                # Runs the forward pass with autocasting.
                #with autocast():
                sum_loss = pointcloud_alignment_model.forward(plot_bool=plot_bool, iteration=i, mesh_for_plotting=mesh_for_plotting)
                print('loss :', sum_loss)
                # Print the losses
                loop.set_description("total_loss = %.6f" % sum_loss)

                if plot_bool:
                    print('plot overall point cloud!!!')

                sum_loss.backward()
                optimizer.step()
        self.losses_results = pointcloud_alignment_model.get_losses
        self.plot_losses()

        r_param_rotation_opt, t_param_translation_opt = pointcloud_alignment_model.get_parameters_rotation_and_translation
        r_param_rotation_opt.requires_grad = False
        t_param_translation_opt.requires_grad = False
        self.scale_xy_plane = pointcloud_alignment_model.get_scale_xy_plane
        self.scale_xy_plane.requires_grad = False

        return r_param_rotation_opt.type(torch.float), t_param_translation_opt.type(torch.float), self.scale_xy_plane

    def plot_losses(self):
        fig_losses = plot_losses(self.losses_results)
        return fig_losses

    @property
    def get_scale_xy_plane(self):
        return self.scale_xy_plane

    @property
    def get_losses(self):
        return self.losses_results

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


def plot_point_clouds_with_error(torch_point_cloud_current,  torch_point_cloud_target_from_mesh  , title:str ='', overall_point_clouds=None, ax=None, fig=None):
    if torch_point_cloud_current.dim() == 3:
        point_cloud_current = torch_point_cloud_current[0].cpu().detach().numpy()
        point_cloud_target_from_mesh =  torch_point_cloud_target_from_mesh[0].cpu().detach().numpy()
    else: print('Incorrect shape!')
    with plt.style.context(('ggplot')):
        if ax is None:
            fig = plt.figure(figsize=(10, 7))
            ax = Axes3D(fig)
        for i in range(point_cloud_current.shape[0]):
            x = np.array((point_cloud_current[i, 0], point_cloud_target_from_mesh[i, 0]))
            y = np.array((point_cloud_current[i, 1], point_cloud_target_from_mesh[i, 1]))
            z = np.array((point_cloud_current[i, 2], point_cloud_target_from_mesh[i, 2]))
            ax.plot(x, y,z, color=red_2, linestyle='-')

        ax.scatter(xs=point_cloud_current[:, 0], ys=point_cloud_current[:, 1], zs=point_cloud_current[:, 2],
                       color=red_2, alpha=0.9, s=35)
        ax.scatter(xs=point_cloud_target_from_mesh[:, 0], ys=point_cloud_target_from_mesh[:, 1], zs=point_cloud_target_from_mesh[:, 2],
                       color=green_4, alpha=0.9, s=35)

        if overall_point_clouds is not None:
            point_cloud_map_from_mesh = overall_point_clouds[0][0].cpu().detach().numpy()
            point_cloud_target_measurment = overall_point_clouds[1][0].cpu().detach().numpy()
            ax.scatter(xs=point_cloud_map_from_mesh[:, 0], ys=point_cloud_map_from_mesh[:, 1], zs=point_cloud_map_from_mesh[:, 2],
                       color=green_6, alpha=0.3, s=1)
            ax.scatter(xs=point_cloud_target_measurment[:, 0], ys=point_cloud_target_measurment[:, 1], zs=point_cloud_target_measurment[:, 2],
                       color=red_3, alpha=0.5, s=1)


        ax.set_title('Scatter plot of pytorch tensor'+title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        return fig, ax


def get_normal_from_spanned_triangle_torch(p1, p2, p3):
    v1 = p3 - p1
    v2 = p2 - p1
    normal = torch.cross(v1, v2)
    return normal