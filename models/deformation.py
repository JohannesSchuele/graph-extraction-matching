import torch
from tqdm.notebook import tqdm
import torch.nn as nn

# io utils

# datastructures

# 3D transformations functions
# rendering components
from configs.plot.config_plots import *
from models.deformation_model import DeformationModel
from models.deformation_model_v2 import DeformationModelTexture
from tools_generate.manage_measurement_data import DepthMeasurementData

class Deformation(nn.Module):
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
    def __init__(self, geometry_mesh, losses, camera, RenderersCollection, device="cpu", image_size=None,
                 world_map=None,  task=None, UpComingDataGenerator=None, use_last_k_images_for_deformation=1,
                 plot_image_nr=0,  depth_map_dim=(128, 128)):
        """
        Parameters device and image size are only used if world_map isn't given
        """
        super().__init__()
        self.world_map = world_map
        if losses:
            self.losses = losses
        else:
            self.losses = {"image reprojection - crossing nodes": {"weight": 1.8, "values": []},
                 "image reprojection - end nodes": {"weight": 0.9, "values": []},
                 "edge": {"weight": 0.4, "values": []},
                 "normal": {"weight": 0.4, "values": []},
                 "laplacian": {"weight": 0.4, "values": []},
                 "chamfer distance to points": {"weight": 0.8, "values": []},
                 "rgb": {"weight": 0.0, "values": []},
                "texture chamfer loss": {"weight": 1.0, "values": []},
                 "pure texture": {"weight": 0.0, "values": []},
                 }

        self.task = {"texture based deformation": False,
                     "texture based chamfer optimization": False,
                     "deformation optimizer SGD": False,
                     "deformation optimizer AdamW": False,
                     "deformation - use Deformation Model Texture": True,
                     "deformation optimizer learning rate": 0.01,
                     "deformation optimizer momentum SGD": 0.9,
                     'deformation - make deforming mesh gif': False,
                     'deformation - visualization of chamfer loss every iteration for gif building': False,
                     'deformation - make gif of mesh model driven by the depth map': False,
                     'deformation - rename losses for LaTeX paper usage': False,
                     'deformation batch size of view for rendering deformed mesh': 1
                     }
        if task is not None:
            self.task.update(task)
        self.UpComingDataGenerator = UpComingDataGenerator
        self.RenderersCollection = RenderersCollection


        self.depth_map_dim =  depth_map_dim
        self.focal_length = camera.focal_length
        self.geometry_mesh = geometry_mesh
        if world_map is not None:
            self.device = world_map.device
            self.image_size = world_map.image_size
        else:
            self.device = device
            self.image_size = image_size

        self.use_last_k_images_for_deformation = use_last_k_images_for_deformation
        self.plot_image_nr = plot_image_nr
        self._ith_call = 0

        self.R_pc_alignment = None
        self.T_pc_alignment = None
        self.scale_xy_plane_alignment = None

        self.loss_dict_history = None
        self.camera_init = camera

        self.texture = None

    def _set_optimizer(self, model):
        #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.7)

        if self.task["deformation optimizer AdamW"]:
            optimizer = torch.optim.AdamW(
                [{'params': model.deform_verts_param, 'lr': self.task["deformation optimizer learning rate"]}] )

        elif self.task["deformation optimizer SGD"]:
            optimizer = torch.optim.SGD(
                [{'params': model.deform_verts_param, 'lr': self.task["deformation optimizer learning rate"]}], momentum=self.task["deformation optimizer momentum SGD"])

        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        parameters_trainable, parameters_all = self.count_parameters(model)
        print('The deformation model, contains ', parameters_trainable, 'trainable parameters and in total ', parameters_all, 'parameters!')
        return optimizer

    def _set_deformation_problem(self, seen_images=None,
                                 data_rotation_translation_batch_lists=None, data_target_imag_batch=None,
                                 data_mask_batch_lists=None, losses=None, world_point_cloud_map=None,
                                 node_point_container_extern=None, depth_map_data=None,
                                 ):
        if losses is not None:
            self.losses = losses
        if self.task["deformation - use Deformation Model Texture"]:

            deformation_model = DeformationModelTexture(geometry_mesh=self.geometry_mesh,
                                                        RenderersCollection=self.RenderersCollection,
                                                        texture=self.texture,
                                                        seen_images=seen_images,
                                                        data_rotation_translation_batch_lists=data_rotation_translation_batch_lists,
                                                        data_target_imag_batch=data_target_imag_batch,
                                                        data_mask_batch_lists=data_mask_batch_lists,
                                                        texture_type=self.RenderersCollection.texture_type,
                                                        losses=self.losses,
                                                        camera_view=self.RenderersCollection.camera_view,
                                                        UpComingDataGenerator=self.UpComingDataGenerator, task=self.task,
                                                        world_point_cloud_map=world_point_cloud_map,
                                                        node_point_container_extern=node_point_container_extern,
                                                        ith_call=self._ith_call,
                                                        )
        else:
            deformation_model = DeformationModel(RenderersCollection=self.RenderersCollection,
                                                 camera_model_view=self.camera_init.clone(),
                                                 geometry_mesh=self.geometry_mesh,
                                                 task=self.task,
                                                 losses=self.losses,
                                                 focal_length=self.focal_length,
                                                 device=self.device,
                                                 image_size=self.image_size,
                                                 plot_image_nr=self.plot_image_nr,
                                                 ith_call=self._ith_call,
                                                 depth_map_data=depth_map_data,
                                                 texture=self.texture,
                                                 )

            if self.task['deformation - depth map optimization'] and self.task['deformation - depth map optimization - align point cloud on mesh']:
                self.R_pc_alignment, self.T_pc_alignment, self.scale_xy_plane_alignment = deformation_model.get_pc_alignment_transformation
                depth_map_data.set_point_cloud_alignment(T=self.T_pc_alignment,
                                                         R=self.R_pc_alignment,
                                                         xy_scale=self.scale_xy_plane_alignment)


        if self.task['deformation - make deforming mesh gif']:
            self.GifMakerWorldMapDeformation = GifMaker(
                name_of_gif='world_map_deforming_mesh_ith_call' + str(self._ith_call))


        return deformation_model, depth_map_data

    def forward_deform(self, geometry_mesh=None,
                       seen_images=None,
                       data_rotation_translation_batch_lists=None,
                       data_target_imag_batch=None, data_mask_batch_lists=None, losses=None, task=None,
                       renderer_settings=None, texture=None,
                       matches_node_pos_order_latest_first=None, target_point_cloud_and_cam_perspective=None,
                       target_image_and_cam_perspective_batch: list=None,
                       world_point_cloud_map=None, node_point_container_extern=None,
                       use_last_k_images_for_deformation=None, depth_map_data=None
                       ):
        self._ith_call += 1
        if geometry_mesh is not None:
            self.geometry_mesh = geometry_mesh
        self.texture = texture
        if task is not None:
            self.task.update(task)
        if use_last_k_images_for_deformation is not None:
            self.use_last_k_images_for_deformation = use_last_k_images_for_deformation
        if renderer_settings is not None:
           self.RenderersCollection.update_renderer(renderer_settings)
        if losses is not None:
            self.losses.update(losses)
        deformation_model, depth_map_data = self._set_deformation_problem(
            seen_images=seen_images,
            data_rotation_translation_batch_lists=data_rotation_translation_batch_lists,
            data_target_imag_batch=data_target_imag_batch,
            data_mask_batch_lists=data_mask_batch_lists, losses=losses,
            world_point_cloud_map=world_point_cloud_map,
            node_point_container_extern=node_point_container_extern,
            depth_map_data=depth_map_data
            )
        optimizer = self._set_optimizer(model=deformation_model)
        # optimization loop:
        loop = tqdm(range(self.task["deformation optimizer number of iterations"]))
        for i in loop:
            # Initialize optimizer
            optimizer.zero_grad()
            plot_bool = i % PLOT_PERIOD_DEFORMATION == 0
            sum_loss, new_src_mesh = deformation_model.forward(plot_bool=plot_bool, iteration=i, ith_call=self._ith_call,
                                                               plot_different_perspectives=False)
            # Print the losses
            loop.set_description("total_loss = %.6f" % sum_loss)
            #print('sum loss : ', sum_loss, 'at iteration: ',i)

            take_image_for_gif = i % MAKE_GIF_WORLD_MAP_DEFORMATION_PERIOD == 0

            if self.task['deformation - make deforming mesh gif']:
                self.make_fig_of_world_map_for_fig(new_src_mesh=new_src_mesh)

            sum_loss.backward(retain_graph=True)
            optimizer.step()
        # visualize_prediction(new_src_mesh, silhouette=True,
        #                      target_image=target_silhouette[1])
        self.loss_dict_history = deformation_model.get_loss_dict_history
        self.plot_losses()
        deformation_model.close_existing_gifs()
        deformation_model.visualize_mesh(title = 'deformed model with texture', batch_size=self.task["deformation batch size of view for rendering deformed mesh"], if_save=True, make_gif=False)
        deformation_model.visualize_texture_prediction(title='deformed model compare texture')

        if self.task['deformation - make deforming mesh gif']:
            self.GifMakerWorldMapDeformation.close_writer()

        return new_src_mesh  # , depth_map_data # Todo: Could be added as return value if data necesary later (may cause conflicts)

    def plot_losses(self, as_log=True):
        fig_plot_losses, ax_losses = plot_losses(losses=self.loss_dict_history,  rename_for_LaTeX=self.task['deformation - rename losses for LaTeX paper usage'] )
        if as_log:
            ax_losses.set_yscale('log', base=10)
        fig_plot_losses.show()
        ax_losses.title.set_visible(False)
        save_figure(fig=fig_plot_losses, name_of_figure='deformation_all_losses_'+str(self._ith_call)+'_ith_call')

    @staticmethod
    def count_parameters(model):
        parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        parameters_all = pytorch_total_params = sum(p.numel() for p in model.parameters())
        return parameters_trainable, parameters_all

    def close_gifs(self):
        if MAKE_GIF_WORLD_MAP_DEFORMATION:
            self.GifMakerWorldMapDeformation.close_writer()

    def update(self, world_map=None, losses=None, mesh=None):

        if world_map is not None:
            self.world_map = world_map
        if losses is not None:
            self.losses = losses
        if mesh is not None:
            self.mesh = mesh

    @property
    def get_latest_pc_alignment_transformation(self):
        return self.R_pc_alignment, self.T_pc_alignment, self.scale_xy_plane_alignment

    def make_fig_of_world_map_for_fig(self, new_src_mesh):
        self.world_map.update_mesh(new_mesh=new_src_mesh)
        fig_plot_world_3D, ax_plot_world_3D = self.world_map.show_scence_from_view(fig=None, ax=None, elev=40, azim=200)

        ax_plot_world_3D.set_xlim([-1, 1])
        ax_plot_world_3D.set_ylim([-1, 1])
        ax_plot_world_3D.set_zlim([-1, 1.5])
        # ax_plot_world_3D.set_xticks([])
        # ax_plot_world_3D.set_yticks([])
        # ax_plot_world_3D.set_zticks([])
        ax_plot_world_3D.set_xlabel(None)
        ax_plot_world_3D.set_ylabel(None)
        ax_plot_world_3D.set_zlabel(None)

        # Turn off tick labels
        ax_plot_world_3D.set_xticklabels([])
        ax_plot_world_3D.set_yticklabels([])
        ax_plot_world_3D.set_zticklabels([])
        #ax_plot_world_3D.axis('off')
        fig_plot_world_3D.tight_layout()
        #fig_plot_world_3D.show()
        self.GifMakerWorldMapDeformation.add_figure(fig=fig_plot_world_3D)


def plot_losses(losses, rename_for_LaTeX=False):
    # fig = plt.figure(figsize=(13, 5))
    # ax = fig.gca()

    if rename_for_LaTeX:
        latex_figure_names = {
        "texture chamfer loss":  '$\mathcal{L}_{\mathsf{chf}}$',
        "pure texture": '$\mathcal{L}_{\mathnormal{tex}}$',
        "edge": '$\mathcal{L}_{\mathnormal{edg}}$',
        "normal":  '$\mathcal{L}_{\mathnormal{nrl}}$',
        "laplacian":  '$\mathcal{L}_{\mathnormal{lap}}$',}
        losses_tmp = losses
        losses= dict((latex_figure_names[key], value) for (key, value) in losses_tmp.items())

    fig, ax = generate_mpl_figure(size_width=3.5, size_height=2.8) #
    for k, l in losses.items():
        is_all_zero = np.all(np.array(l['values']) == 0.0)
        if not is_all_zero:
            if 'loss' in k or rename_for_LaTeX is True:
                ax.plot(l['values'], label=k)
            else:
                ax.plot(l['values'], label=k + " loss")


    ax.legend(loc='upper right')
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Loss vs Iterations")

    #ax.get_legend().remove() #ToDo make this freely accessable
    fig.tight_layout()
    fig.show()
    return fig, ax


