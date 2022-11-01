import os
import torch
import numpy as np
from tqdm.notebook import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights, AmbientLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesAtlas,
    TexturesVertex
)
from models.renderer import MeshRendererWithFragments2PointCloud
from models.renderer import SoftMesh2Image2PointCloudShader
from utils.texture_visualization import visualize_mesh
from configs.plot.config_plots import *
from enum import Enum
class TextureTypes(Enum):
    VERTEX_TEXTURE = 'Vertex Texture' #ToDo may delete that, since it shouldn't be used!
    ATLAS_TEXTURE = 'Atlas Texture'
    VERTEX_TEXTURE_RGB = 'Vertex Texture RGB'
    VERTEX_TEXTURE_GREY = 'Vertex Texture Grey'
from models.texture_model import TextureModel
from pytorch3d.renderer import BlendParams

class InitRenderers(nn.Module):
    def __init__(self, camera_texture, image_size=None, camera_view=None, renderer_settings=None, device='cpu'):

        super().__init__()
        self.device = device
        if image_size is not None:
            self.image_size = image_size
        elif camera_texture.image_size is not None:
            self.image_size = int(camera_texture.image_size.clone().detach().cpu().numpy()[0][0])
        else:
            print('Image size needs to be specified for the rendering functions!')
            self.image_size = 128
        self.focal_length = camera_texture.focal_length
        self.principal_point = camera_texture.principal_point
        self.renderer_settings = {"face per pixel - opt": 4,
                                  "blur radius - opt": np.log(1. / 1e-4 - 1.)*1e-5,
                                  "blend param sigma - opt": 1e-5,
                                  "blend param gamma - opt": 5*1e-8,
                                  "face per pixel - high opt": 25,
                                  "blur radius - high opt": 1*np.log(1. / 1e-4 - 1.)*1e-5,
                                  "blend param sigma - high opt": 1*1e-5,
                                  "blend param gamma - high opt": 1 * 1e-7,

                                  "face per pixel - view": 1,
                                  "blur radius - view": 1e-7,
                                  "blend param sigma - view": None,
                                  "blend param gamma - view": None,
                                  "face per pixel - inverse": 20,
                                  "blur radius - inverse": 2 * np.log(1. / 1e-4 - 1.)*1e-5,
                                  "blend param sigma - inverse": 2* 1e-5,
                                  "blend param gamma - inverse": 1*1e-5,
                                  "image size view rendering": 256,

                                  "face per pixel - inverse nodePoseOpt": 40,
                                  "blur radius - inverse nodePoseOpt": np.log(1. / 1e-4 - 1.) * 1e-5,
                                  "blend param sigma - inverse nodePoseOpt": 1*1e-4,
                                  "blend param gamma - inverse nodePoseOpt": 1e-5,
                                  }

        if renderer_settings is not None:
            self.renderer_settings.update(renderer_settings)

        self.camera_texture = camera_texture
        self.init_renders(camera_view=camera_view)
        self.init_inverse_renderer_nodePoseOpt()


    def init_renders(self, camera_view = None):
        perspective_correct = False
        if camera_view is not None:
            self.camera_view = camera_view
        else:
            self.camera_view = PerspectiveCameras(device=self.device, R=self.camera_texture.R.clone(), T=self.camera_texture.T.clone(),
                                                  focal_length=self.camera_texture.focal_length* 0.5, principal_point=self.camera_texture.principal_point)

        self.raster_settings_view = RasterizationSettings(image_size=self.renderer_settings["image size view rendering"], blur_radius=self.renderer_settings["blur radius - view"], faces_per_pixel=self.renderer_settings["face per pixel - view"], perspective_correct = perspective_correct, clip_barycentric_coords = None, cull_backfaces = True,)
        self.raster_settings_texture = RasterizationSettings(image_size=self.image_size, blur_radius=self.renderer_settings["blur radius - opt"], faces_per_pixel=self.renderer_settings["face per pixel - opt"], perspective_correct = perspective_correct, clip_barycentric_coords = None, cull_backfaces = True,)

        self.lights_view_point_lights = PointLights(device=self.device, location=[[2.0, 2.0, 5.0]])
        self.lights_ambient = AmbientLights(device = self.device)

        self.materials_texture = Materials(device=self.device, ambient_color=[[1, 1, 1]], diffuse_color=[[0.0, 0.0, 0.0]], specular_color=[[0.0, 0.0, 0.0]], shininess=0.0)
        self.materials_view = Materials(device=self.device, specular_color=[[0.99, 0.99, 0.99]], ambient_color=[[1, 1, 1]], shininess=9)#1 for the rest

        self.renderer_view = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.camera_view, raster_settings=self.raster_settings_view),
            shader=SoftPhongShader(device=self.device, cameras=self.camera_view, lights=self.lights_view_point_lights, materials=self.materials_view))

        self.renderer_texture = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.camera_texture, raster_settings=self.raster_settings_texture),
            shader=SoftPhongShader(device=self.device, cameras=self.camera_texture, lights=self.lights_ambient, materials=self.materials_texture, blend_params= BlendParams(sigma=self.renderer_settings["blend param sigma - opt"], gamma=self.renderer_settings["blend param gamma - opt"])))

        self.raster_settings_texture_high_diff = RasterizationSettings(image_size=self.image_size,
                                                             blur_radius=self.renderer_settings["blur radius - high opt"],
                                                             faces_per_pixel=self.renderer_settings[
                                                                 "face per pixel - high opt"], perspective_correct=perspective_correct,
                                                             clip_barycentric_coords=None, cull_backfaces=True, )
        self.renderer_texture_high_diff = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.camera_texture, raster_settings=self.raster_settings_texture),
            shader=SoftPhongShader(device=self.device, cameras=self.camera_texture, lights=self.lights_ambient,
                                   materials=self.materials_texture,
                                   blend_params=BlendParams(sigma=self.renderer_settings["blend param sigma - high opt"],
                                                            gamma=self.renderer_settings["blend param gamma - high opt"])))

        raster_settings_soft = RasterizationSettings( image_size=self.image_size, bin_size=20, blur_radius= self.renderer_settings["blur radius - inverse"], faces_per_pixel=self.renderer_settings["face per pixel - inverse"])
        self.inverse_renderer = MeshRendererWithFragments2PointCloud(
                                        rasterizer=MeshRasterizer(cameras=self.camera_texture, raster_settings=raster_settings_soft),
                                        shader = SoftMesh2Image2PointCloudShader(cameras=self.camera_texture, blend_params = BlendParams(sigma=self.renderer_settings["blend param sigma - inverse"], gamma=self.renderer_settings["blend param gamma - inverse"])))
                                            # gamma->0, the aggregation function only outputs the color of the nearest triangle,
                                            # which exactly matches the behavior of z-buffering.

        if "texture type for reconstruction" in self.renderer_settings:
            self.texture_type = self.renderer_settings["texture type for reconstruction"]
        else:
            self.texture_type = TextureTypes.VERTEX_TEXTURE_RGB

    def init_inverse_renderer_nodePoseOpt(self):

        raster_settings_soft_inverse_renderer_nodePoseOpt = RasterizationSettings(image_size=self.image_size, bin_size=40,
                                                                                  blur_radius=self.renderer_settings["blur radius - inverse nodePoseOpt"],
                                                                                  faces_per_pixel=self.renderer_settings["face per pixel - inverse nodePoseOpt"],
                                                                                  perspective_correct=True,
                                                                                  clip_barycentric_coords=None,
                                                                                  cull_backfaces=True,
                                                                                  )
        self.inverse_renderer_nodePoseOpt = MeshRendererWithFragments2PointCloud(
            rasterizer=MeshRasterizer(cameras=self.camera_texture, raster_settings=raster_settings_soft_inverse_renderer_nodePoseOpt),
            shader=SoftMesh2Image2PointCloudShader(cameras=self.camera_texture, blend_params=BlendParams(
                sigma=self.renderer_settings["blend param sigma - inverse nodePoseOpt"],
                gamma=self.renderer_settings["blend param gamma - inverse nodePoseOpt"])))

    def update_renderer(self, renderer_settings, camera_texture=None, camera_view=None, update_inverse_renderer_nodePoseOpt=False):
        if renderer_settings is not None:
            self.renderer_settings.update(renderer_settings)
        if camera_view is not None:
            self.camera_view = camera_view
        if camera_texture is not None:
            self.camera_texture = camera_texture
        self.init_renders(camera_view=camera_view)
        if update_inverse_renderer_nodePoseOpt:
            self.init_inverse_renderer_nodePoseOpt()

    @property
    def image_size_plain(self):
        return self.image_size

#    @property
#    def image_size_torch_tensor(self, device=None):
#        if device is not None:
#            return torch.tensor([self.image_size, self.image_size], device=device)
#        else:
#            return torch.tensor([self.image_size, self.image_size], device=self.device)





class Texture(nn.Module):
    def __init__(self, mesh, camera_texture,
                 losses=None, task=None, UpComingDataGenerator =None, RenderersCollection=None, renderer_settings=None, geometry_mesh=None, device='cpu'):
        super().__init__()
        self.device = device
        self.camera_texture = camera_texture
        self.UpComingDataGenerator = UpComingDataGenerator

        if losses is not None:
            self.losses = losses
        else:
            self.losses = {"chamfer single iteration loss": {"weight": 1.0, "values": []},
                           "chamfer every iteration loss": {"weight": 1.0, "values": []},
                           "image pattern loss": {"weight": 1.0, "values": []},
                           "texture reconstruction loss": {"weight": 1.0, "values": []}}
        self.task = {"texture learning": True,
                     "pose by texture": True,
                     "fix first pose": True,
                     "use chamfer loss": False,
                     "use chamfer loss - make plots": False,
                     "texture type for reconstruction": TextureTypes.VERTEX_TEXTURE_GREY,
                     "batch size for rendering": 5,
                     "number of last images 2b used": 10,
                     "learning rate - texture": 1e-2,
                     "learning rate - pose": 1e-1,
                     'Texture Model - batch size of mesh visualization': 1,

                     'Texture optimization - shift by an grey value range: float between zero and one': 0.1,
                     'use chamfer loss - plot period for every-ith iteration': 1,
                     'Pose optimization based on chamfer loss using 3D projection!': False,
                     'Pose optimization based on chamfer loss using 3D projection - Plot point cloud on mesh ever-ith iteration!':10,
                     'Pose optimization based on chamfer loss using 3D projection - Plot point cloud on mesh!': True,
                     'Threshold texture after grey scale texture optimization - threshold as float': 0.5,
                     'Threshold texture after grey scale texture optimization': False,
                     'Texture Optimization - plot losses in paper naming using LaTeX font':False,
                     }

        if task is not None:
            self.task.update(task)
        if isinstance(RenderersCollection, InitRenderers):
            self.RenderersCollection = RenderersCollection
            if renderer_settings is not None:
                self.RenderersCollection.update_renderer(renderer_settings=renderer_settings)
        else:
            self.RenderersCollection = InitRenderers(camera_texture=camera_texture, image_size=camera_texture.image_size, camera_view=None, renderer_settings=renderer_settings, device=device)

        if "texture type for reconstruction" in renderer_settings:
            self.texture_type = self.task["texture type for reconstruction"]
        else:
            self.texture_type = TextureTypes.VERTEX_TEXTURE_RGB
        #Todo: check if a texture is already provided through the given mesh!
        self._init_texture(mesh=mesh)
        self.TextureModel = None
        self.seen_images = None
        self.ith_call = 0
        self.geometry_mesh = geometry_mesh

    def _init_texture(self, mesh):
        self.mesh_texture = mesh.clone()
        del mesh
        if self.texture_type == TextureTypes.ATLAS_TEXTURE:
            """
            A texture representation where each face has a square texture map.
                atlas: (N, F, R, R, C) tensor giving the per face texture map.
                    The atlas can be created during obj loading with the
                    pytorch3d.io.load_obj function - in the input arguments
                    set `create_texture_atlas=True`. The atlas will be
                    returned in aux.texture_atlas.
            The padded and list representations of the textures are stored
            and the packed representations is computed on the fly and
            not cached.
                See also https://github.com/ShichenLiu/SoftRas/issues/21
            """
            atlas_dim_per_face = 25
            face_shape = self.mesh_texture.faces_packed().shape
            rgb_atlas = torch.full([1, face_shape[0], atlas_dim_per_face, atlas_dim_per_face, 3], 0.3, device=self.device, requires_grad=False)
            self.mesh_texture.textures = TexturesAtlas(rgb_atlas)
        elif self.texture_type == TextureTypes.VERTEX_TEXTURE or self.texture_type == TextureTypes.VERTEX_TEXTURE_RGB:
            """
            Batched texture representation where each vertex in a mesh
            has a C dimensional feature vector.
            Args:
                verts_features: list of (Vi, C) or (N, V, C) tensor giving a feature
                    vector with arbitrary dimensions for each vertex.
            """
            verts_shape = self.mesh_texture.verts_packed().shape
            sphere_verts_rgb = torch.full([1, verts_shape[0], 3], self.task['Texture optimization - shift by an grey value range: float between zero and one'], device=self.device, requires_grad=False)
            self.mesh_texture.textures = TexturesVertex(verts_features=sphere_verts_rgb)

        elif self.texture_type == TextureTypes.VERTEX_TEXTURE_GREY:
            """
            Batched texture representation where each vertex in a mesh
            has a C dimensional feature vector.
            Args:
                verts_features: list of (Vi, C) or (N, V, C) tensor giving a feature
                    vector with arbitrary dimensions for each vertex.
            """

            verts_shape = self.mesh_texture.verts_packed().shape
            sphere_verts_rgb = torch.full([1, verts_shape[0], 1], self.task['Texture optimization - shift by an grey value range: float between zero and one'], device=self.device, requires_grad=False)
            self.mesh_texture.textures = TexturesVertex(verts_features=sphere_verts_rgb)

    def _set_optimizer(self, model, lr = 0.1):
        # The optimizer
        #https://pytorch.org/docs/stable/optim.html
        if model.texture_learning_only:
            optimizer = torch.optim.AdamW(
                [{'params': model.sphere_verts_rgb, 'lr': self.task["learning rate - texture"]}], lr=lr)

        elif model.pose_estimation_only:
            optimizer = torch.optim.Adam([{'params': model.r_rotation_batch_param, 'lr': self.task["learning rate - pose"]},
                                          {'params': model.t_translation_batch_param, 'lr': self.task["learning rate - pose"]}], lr=lr)
        elif model.texture_and_pose_learning:
            optimizer = torch.optim.AdamW([{'params': model.r_rotation_batch_param, 'lr': self.task["learning rate - pose"]},
                                          {'params': model.t_translation_batch_param, 'lr': self.task["learning rate - pose"]},
                                          {'params': model.sphere_verts_rgb, 'lr': self.task["learning rate - texture"]}], lr=lr)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        parameters_trainable, parameters_all = self.count_parameters(model)
        print('The texture model has', parameters_trainable, 'trainable parameters and in total ', parameters_all, 'parameters!')
        return optimizer

    def _set_texture_problem(self, seen_images, target_images:torch.Tensor=None, image_mask=None, target_cameras=None,world_point_cloud_map=None, node_point_container_extern=None):
        model_texture = TextureModel(mesh=self.mesh_texture, RenderersCollection=self.RenderersCollection,
                                     seen_images=seen_images,
                                     target_images = target_images, loaded_target_cameras = target_cameras,
                                     image_mask=image_mask,
                                     losses=self.losses,
                                     camera_view=self.RenderersCollection.camera_view,
                                     UpComingDataGenerator=self.UpComingDataGenerator, task=self.task,
                                     world_point_cloud_map=world_point_cloud_map,
                                     node_point_container_extern=node_point_container_extern,
                                     geometry_mesh = self.geometry_mesh,
                                     ith_call =self.ith_call
                                     )
        self.seen_images = seen_images
        return model_texture

    def forward(self, seen_images=None, last_images_2bused=None,
                target_images:torch.Tensor=None, target_cameras=None, image_mask=None, losses=None, task=None, renderer_settings=None, mesh_texture=None,
                world_point_cloud_map=None, node_point_container_extern=None, iterations = 55, lr=0.3):
        # if target_images and /or target_cameras are None, the data is loaded through the DataGenerator\\
        # from already seen_images!
        assert not(seen_images is None and target_images is None)
        assert not(seen_images is None and target_cameras is None)
        assert (self.UpComingDataGenerator is not None and seen_images is not None) or (target_images is not None and target_cameras is not None)
        if mesh_texture is not None:
            self.mesh_texture = mesh_texture
        self.update_renderer(renderer_settings)
        self.update_losses(losses=losses)
        self.update_task(task=task)
        self.ith_call = self.ith_call + 1
        if self.task["use chamfer loss - make plots"]:
            self.Gif_chamfer_loss = GifMaker(name_of_gif='Chamfer_loss_pose_reconstruction_' + str(self.ith_call))

        self.TextureModel = self._set_texture_problem(seen_images=seen_images, target_images=target_images, image_mask=image_mask, target_cameras=target_cameras,  world_point_cloud_map=world_point_cloud_map,
                                     node_point_container_extern=node_point_container_extern)
        optimizer = self._set_optimizer(model=self.TextureModel, lr=lr)
        # optimization loop:
        loop = tqdm(range(iterations))
        for i in loop:

            with torch.autograd.set_detect_anomaly(True):
                # Initialize optimizer
                optimizer.zero_grad()
                sum_loss = self.TextureModel.forward(iteration=i)
                # Print the losses
                loop.set_description("total_loss = %.6f" % sum_loss)

                #print('Rotation and Translation: ', self.TextureModel.get_rotation_translation())

                if i % PLOT_PERIOD_TEXTURE == 0:
                    if PLOT_TEXTURE_COMPARE_AND_MESH:
                        self.TextureModel.visualize_texture_prediction(title='iter: %d texture fitting vs. target image' % i)
                        self.TextureModel.visualize_mesh(title='iter: %d Model with texture' % i)

                    if self.task["use chamfer loss - make plots"] and self.TextureModel.fig_ax_chamfer_loss_pcs_pose_recon is not None and i % self.task["use chamfer loss - plot period for every-ith iteration!"] == 0:
                        return_texture_figure = self.TextureModel.fig_ax_chamfer_loss_pcs_pose_recon
                        if return_texture_figure is not None:
                            fig_chamfer_pcs_pose_recon, ax_chamfer_pcs_pose_recon = return_texture_figure
                            ax_chamfer_pcs_pose_recon.title.set_visible(False)
                            fig_chamfer_pcs_pose_recon, ax_chamfer_pcs_pose_recon = keep_grid_and_remove_ticks_and_labels(
                                fig=fig_chamfer_pcs_pose_recon, ax=ax_chamfer_pcs_pose_recon)
                            save_figure(fig=fig_chamfer_pcs_pose_recon, ax=ax_chamfer_pcs_pose_recon, show_fig=True, name_of_figure='Chamfer_loss_point_cloud_alignment_on_rendered_image_call' +str(self.ith_call) +'_itr_'+str(i))
                            ax_chamfer_pcs_pose_recon.axis('off')
                            self.Gif_chamfer_loss.add_figure(fig=fig_chamfer_pcs_pose_recon, ax=ax_chamfer_pcs_pose_recon)

                sum_loss.backward(retain_graph=False)
                optimizer.step()

        self.TextureModel.visualize_texture_prediction(title='texture fitting vs. target image final solution')
        #self.TextureModel.visualize_mesh(title='Model with texture final solution')

        #ToDo to be implemented!!!!
        #self.TextureModel.visualize_mesh_over_batch(title='Model with texture final solution', batch_size=self.task['Texture Model - batch size of mesh visualization'])

        self.plot_losses(as_log=True)
        if self.task["use chamfer loss - make plots"]:
            self.Gif_chamfer_loss.close_writer()

        new_textured_mesh = self.TextureModel.get_textured_mesh()
        if self.task['Threshold texture after grey scale texture optimization'] and self.task["texture learning"]:
            new_textured_mesh = self.get_thresholded_mesh(textured_mesh=new_textured_mesh)

        # overwrite the outdated textured mesh with the solution
        self.mesh_texture = new_textured_mesh
        return new_textured_mesh

    def get_thresholded_mesh(self, textured_mesh):
        if self.texture_type is TextureTypes.VERTEX_TEXTURE_GREY and self.task['Threshold texture after grey scale texture optimization']:
            feature_threshold_for_robustness_optimization = torch.nn.Threshold(threshold=self.task['Threshold texture after grey scale texture optimization - threshold as float'], value=self.task['Texture optimization - shift by an grey value range: float between zero and one'])

            mesh_texture_thresholded = feature_threshold_for_robustness_optimization(
                textured_mesh.textures.verts_features_padded())
            textured_mesh.textures = TexturesVertex(verts_features=mesh_texture_thresholded)

            if self.task['Threshold texture after grey scale texture optimization - plot thresholded model']:
                fig, ax = visualize_mesh(textured_mesh=textured_mesh, renderer=self.get_RenderersCollection.renderer_view, camera_perspective=self.get_RenderersCollection.camera_view, title='Model after thresholding',
                                   silhouette=False, grey=False, size_width=3.5, size_height=3.5)
                #ToDo save figure!
                fig.show()

        return textured_mesh


    def plot_losses(self, as_log=False, size_width=3.45, size_height=1.8):
        if self.TextureModel is not None:
            fig_plot_losses, ax_losses = plot_losses(losses=self.TextureModel.get_loss_dict_history, size_width=size_width, size_height=size_height, rename_for_LaTeX=self.task['Texture Optimization - plot losses in paper naming using LaTeX font'] )
            if as_log:
                ax_losses.set_yscale('log', base=10)
            fig_plot_losses.show()
            ax_losses.title.set_visible(False)
            save_figure(fig=fig_plot_losses, name_of_figure='texture_based_reconstruction_all_losses_' + str(self.ith_call) + '_ith_call')

    def update_losses(self, losses):
        if losses is not None:
            self.losses = losses
        if self.TextureModel is not None:
            self.TextureModel.update_losses(losses=losses)

    def update_renderer(self, renderer_settings, camera_view=None):
        if renderer_settings is not None:
            self.RenderersCollection.update_renderer(renderer_settings, camera_view=camera_view)

    def update_task(self, task):
        if task is not None:
            self.task.update(task)
        if self.TextureModel is not None:
            self.TextureModel.update_task(task=task)

    @property
    def get_textured_mesh(self):
        return self.mesh_texture

    @property
    def get_RenderersCollection(self):
        return self.RenderersCollection
    # for param in self.parameters():
    #     print(type(param), param.size())
    @staticmethod
    def count_parameters(model):
        parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        parameters_all = pytorch_total_params = sum(p.numel() for p in model.parameters())
        return parameters_trainable, parameters_all

    @ property
    def get_idx_of_last_image_applied(self):
        return self.self.TextureModel.get_idx_of_last_image_applied

    def get_updated_cam_positions_in_seen_images(self):
        if self.seen_images is not None:
            r_rotation_batch, t_translation_batch = self.TextureModel.get_rotation_translation()
            r_rotation_batch.requires_grad = False # not necessary here?!
            t_translation_batch.requires_grad = False
            last_images_2bused = r_rotation_batch.shape[0]
            for i, graph_image_s in enumerate(self.seen_images[: last_images_2bused]):
                graph_image = graph_image_s[2]
                graph_image.update_cam_position(r_rotation_batch[i], t_translation_batch[i])
                self.seen_images[i][2] = graph_image
            return self.seen_images
        else:
            return False

    def get_rotation_translation_parameters(self):
        r_rotation_batch, t_translation_batch = self.TextureModel.get_rotation_translation()
        r_rotation_batch.requires_grad = False # not necessary here?!
        t_translation_batch.requires_grad = False
        return r_rotation_batch, t_translation_batch

    @ property
    def get_seen_images(self):
        return self.seen_images

# Plot losses as a function of optimization iteration
# def plot_losses(losses):
#     fig = plt.figure(figsize=(13, 5))
#     ax = fig.gca()
#     for k, l in losses.items():
#         ax.plot(l['values'], label=k + " loss")
#     ax.legend(fontsize="16")
#     ax.set_xlabel("Iteration", fontsize="16")
#     ax.set_ylabel("Loss", fontsize="16")
#     ax.set_title("Loss vs iterations", fontsize="16")
#     plt.show()
def plot_losses(losses,size_width=3.45, size_height=2.45, rename_for_LaTeX=False):
    # fig = plt.figure(figsize=(13, 5))
    # ax = fig.gca()
    if rename_for_LaTeX:
        latex_figure_names = {
            "texture reconstruction loss": '$\mathcal{L}_{\mathnormal{tex}}$',
            "image pattern loss": '$\mathcal{L}_{\mathnormal{tex}}$',
             "chamfer every iteration loss": '$\mathcal{L}_{\mathnormal{chf}}$',
            "chamfer single iteration loss": '$\mathcal{L}_{\mathnormal{chf}}$',
            "chamfer loss 3d data": '$\mathcal{L}_{\mathnormal{chf}}$',
            'pose chamfer loss': '$\mathcal{L}_{\mathnormal{chf}}$',
        }
        #losses = dict((latex_figure_names[key], value) for (key, value) in losses_tmp.items())
        losses_tmp = {}
        for key, value in losses.items():
            is_all_zero = np.all(np.array(value['values']) == 0.0)
            if not is_all_zero:
                losses_tmp.update({latex_figure_names[key]: value})

        losses = losses_tmp

    fig, ax = generate_mpl_figure(size_width=size_width, size_height=size_height) #
    for k, l in losses.items():
        is_all_zero = np.all(np.array(l['values']) == 0.0)
        scale = 0.1

        if not is_all_zero:
            scaled_values = np.array(l['values'])*scale
            if rename_for_LaTeX is True:
                ax.plot(scaled_values, label=k)
            else:
                if 'single iteration' in k:
                    ax.plot(scaled_values, label='pose chamfer loss')
                elif 'loss' in k:
                    ax.plot(scaled_values, label=k)
                else:
                    ax.plot(scaled_values, label=k + " loss")
    ax.legend(loc='upper right')
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Loss vs Iterations")

    #ax.get_legend().remove() #ToDo make this freely accessable
    fig.tight_layout()
    return fig, ax