import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
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
from utils.texture_visualization import *
from models.pose_rodrigues_rot_fromula import Camera3DPoseRodriguesRotFormula
from models.texture import TextureTypes
from configs.config import *
from pytorch3d.renderer.cameras import (
    PerspectiveCameras)
from configs.plot.config_plots import *
from tools_generate.mask import plot_image_with_mask
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from utils.plot_images import visualize_image
from models.multiview import CameraTorch, calc_reprojection_error_matrix, \
    project_3d_points_to_image_plane_without_distortion
from utils.deformation_helpers import NodePointContainer
from tools_generate.mask import generate_mask
from configs.plot.colours import *
from models.camera_pose import camera_projection, plot_img_2d_points
from pytorch3d.utils import opencv_from_cameras_projection
class TextureModel(nn.Module):
    """
       A texture representation where each face has a square texture map.
       This is based on the implementation from SoftRasterizer [1].

       Args:
           atlas: (N, F, R, R, C) tensor giving the per face texture map.
               The atlas can be created during obj loading with the
               pytorch3d.io.load_obj function - in the input arguments
               set `create_texture_atlas=True`. The atlas will be
               returned in aux.texture_atlas.


       The padded and list representations of the textures are stored
       and the packed representations is computed on the fly and
       not cached.

       [1] Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based
           3D Reasoning', ICCV 2019
           See also https://github.com/ShichenLiu/SoftRas/issues/21
       """
    def __init__(self, mesh, seen_images, RenderersCollection, UpComingDataGenerator=None,
                 camera_view=None, target_images:torch.Tensor=None, image_mask=None, world_point_cloud_map=None, node_point_container_extern=None,
                 loaded_target_cameras=None, losses=None, task=None, geometry_mesh=None, ith_call=0):
        super().__init__()
        self.camera_view = camera_view
        self.device = mesh.device
        self.ith_call = ith_call

        self.seen_images = seen_images
        self.renderer_texture_observe = RenderersCollection.renderer_texture
        self.renderer_texture_high_diff = RenderersCollection.renderer_texture_high_diff
        self.focal_length = RenderersCollection.focal_length
        self.renderer_view = RenderersCollection.renderer_view
        self.inverse_renderer = RenderersCollection.inverse_renderer
        self.lights_view_point_lights = RenderersCollection.lights_view_point_lights
        self.lights_ambient = RenderersCollection.lights_ambient

        self.image_size_torch_tensor = torch.tensor([RenderersCollection.image_size, RenderersCollection.image_size], device=self.device)
        self.image_size_int = RenderersCollection.image_size
        if losses is not None:
            self.losses = losses
        else:
            self.losses = {"chamfer single iteration loss": {"weight": 1.0, "values": []},
                               "chamfer every iteration loss": {"weight": 1.0, "values": []},
                               "image pattern loss": {"weight": 1.0, "values": []},
                               "texture reconstruction loss": {"weight": 1.0, "values": []},
                                "chamfer loss 3d data": {"weight": 10.0, "values": []},
            }

        self.task = {"texture learning": True,
                       "pose by texture": True,
                     "fix first pose": True,
                     "use chamfer loss - make plots": False,
                     "use chamfer loss - make plots - all variations": False,
                     "texture use world map node points for masking chamfer pcs": False,
                     "texture type for reconstruction": TextureTypes.VERTEX_TEXTURE_GREY,
                     "use chamfer loss with single point cloud extraction": False,
                     "use chamfer loss with point cloud extraction over every iteration": False,
                     'Texture Reconstruction - load input data from seen_images else function parameters have to be set!':True,
                     "use chamfer loss": False,
                     "use image pattern loss for pose": False,
                     "batch size for rendering": 5,
                     "number of last images 2b used": 10,
                     "learning rate - texture": 1e-2,
                     "learning rate - pose": 1e-1,
                     'Pose optimization based on chamfer loss using 3D projection!':False,
                     'Texture optimization - shift by an grey value range: float between zero and one': 0.1,
                     'Threshold texture for grey scale texture optimization': False,
                     }

        if task is not None:
            self.task.update(task)
        if self.task["use chamfer loss"]:
            self.idx_matrix = self.get_index_matrix(image_size=self.image_size_int, device=self.device)

        self.last_images_2bused = self.task["number of last images 2b used"] # last_images_2bused must be at least >=2 if only camera pose it to be optimized! -> else there are no parameters!
            #just one at the sime time
        self.pose_estimation_only = self.task["pose by texture"] and self.task["texture learning"] is False
        self.texture_learning_only = self.task["texture learning"] and self.task["pose by texture"] is False
        self.texture_and_pose_learning = self.task["texture learning"] and self.task["pose by texture"] is True
        self.fig_ax_chamfer_loss_pcs_pose_recon = None
        self.UpComingDataGenerator = UpComingDataGenerator
        self.image_grey_scale_shift = None


        if "texture type for reconstruction" in self.task:
            self.texture_type = self.task["texture type for reconstruction"]
        else:
            self.texture_type = TextureTypes.VERTEX_TEXTURE_RGB

        if self.task["Pose optimization based on chamfer loss using 3D projection!"]:
            self.feature_threshold_for_robustness = torch.nn.Threshold(threshold=0.6, value=0.0)


        self.image_mask = image_mask
        if self.seen_images is not None and self.task['Texture Reconstruction - load input data from seen_images else function parameters have to be set!']:
            r_rotation_batch, t_translation_batch = self._get_poses_target_images(target_images=target_images, world_point_cloud_map=world_point_cloud_map)
        else:
            r_rotation_batch = None
            t_translation_batch = None
            if target_images is not None:
                if self.texture_type == TextureTypes.VERTEX_TEXTURE_GREY:
                    target_image_grey = target_images
                    self.target_images = torch.stack((target_image_grey, target_image_grey, target_image_grey), dim=3)
                else:
                    self.target_images = target_images
            if self.task["texture use world map node points for masking chamfer pcs"]:
                self.node_point_container = node_point_container_extern

        if self.texture_learning_only:
            self._define_texture_parameters(mesh=mesh, texture_type=self.texture_type)
            self._parametrize_pose_batches_2lists(r_rotation_batch, t_translation_batch)
            # self.r_rotation_batch = r_rotation_batch
            # self.t_translation_batch = t_translation_batch
        elif self.pose_estimation_only:
            self._parametrize_pose_batches_2lists(r_rotation_batch, t_translation_batch)
            #self._define_texture(mesh=mesh, texture_type=texture_type)
            self.mesh_texture = mesh
            self.geometry_mesh = geometry_mesh

        else: # both is optimized at the same time!
            self._define_texture_parameters(mesh=mesh, texture_type=self.texture_type)
            self._parametrize_pose_batches_2lists(r_rotation_batch, t_translation_batch)

        self.param_r_rotation_batch = None
        self.param_t_translation_batch = None
        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=self.device)
        self.target_camera_perspectives = loaded_target_cameras
        self.fig_chamfer_loss_pcs_pose_recon = None


    def _define_texture_parameters(self, mesh, texture_type:TextureTypes):
        # self.mesh_texture = mesh.clone()
        # del mesh
        self.mesh_texture = mesh
        if texture_type is TextureTypes.ATLAS_TEXTURE:
            # parameter must be stored/ (kept as an attribute) in the nn.Module class!
            self.rgb_atlas_param = nn.Parameter(self.mesh_texture.textures.atlas_padded().clone().requires_grad_(True))
            #rgb_atlas_param = nn.Parameter(self.mesh_texture.textures.atlas_packed().clone().requires_grad_(True))
            self.mesh_texture.textures = TexturesAtlas(self.rgb_atlas_param)
        elif texture_type is TextureTypes.VERTEX_TEXTURE or texture_type is TextureTypes.VERTEX_TEXTURE_RGB or texture_type is TextureTypes.VERTEX_TEXTURE_GREY:
            # learn per vertex colors for our sphere mesh that define texture
            self.sphere_verts_rgb = nn.Parameter(self.mesh_texture.textures.verts_features_padded().clone().requires_grad_(True))
            self.mesh_texture.textures = TexturesVertex(verts_features=self.sphere_verts_rgb)

        self.r_rotation_batch_param, self.t_translation_batch_param = None, None

    # def _define_texture(self, mesh, texture_type:TextureTypes):
    #     self.mesh_texture = mesh
    #     if texture_type is TextureTypes.ATLAS_TEXTURE:
    #         # parameter must be stored/ (kept as an attribute) in the nn.Module class!
    #         self.rgb_atlas = self.mesh_texture.textures.atlas_padded().clone()
    #         self.mesh_texture.textures = TexturesAtlas(self.rgb_atlas_param)
    #     elif texture_type is TextureTypes.VERTEX_TEXTURE or texture_type is TextureTypes.VERTEX_TEXTURE_RGB:
    #         self.sphere_verts_rgb = self.mesh_texture.textures.verts_features_padded().clone()
    #         self.mesh_texture.textures = TexturesVertex(verts_features=self.sphere_verts_rgb)

    def _parametrize_pose_batches_2lists(self, r_rotation_batch_all, t_translation_batch_all):
        if  r_rotation_batch_all is not None and t_translation_batch_all is not None:
            if self.task["pose by texture"] and self.task["fix first pose"]:
                self.r_rotation_batch_fixed = r_rotation_batch_all[-1:] # last entry in list corresponds to initial cam position
                self.t_translation_batch_fixed = t_translation_batch_all[-1:]
                # just applicable if at least >=2 images were already seen
                self.r_rotation_batch_param = nn.Parameter(r_rotation_batch_all[0:-1]) # first entry corresponds to latest cam observation!
                self.t_translation_batch_param = nn.Parameter(t_translation_batch_all[0:-1])

            if self.task["pose by texture"] and self.task["fix first pose"] == False:
                self.r_rotation_batch_fixed = None
                self.t_translation_batch_fixed = None
                # just applicable if at least >=2 images were already seen
                self.r_rotation_batch_param = nn.Parameter(r_rotation_batch_all)
                self.t_translation_batch_param = nn.Parameter(t_translation_batch_all)

            elif self.texture_learning_only:
                self.r_rotation_batch_fixed = r_rotation_batch_all
                self.t_translation_batch_fixed = t_translation_batch_all
                self.r_rotation_batch_param = None
                self.t_translation_batch_param = None

        else:
            self.r_rotation_batch_fixed = r_rotation_batch_all
            self.t_translation_batch_fixed = t_translation_batch_all

    def _get_poses_target_images(self, target_images=None, world_point_cloud_map=None):
        # initialize the absolute log-rotations/translations with random-zero entries

        if self.last_images_2bused == 'all':
            self.last_images_applied = self.seen_images.__len__()
        else:
            self.last_images_applied = min(self.last_images_2bused, self.seen_images.__len__())

        r_rotation_batch = torch.zeros((self.last_images_applied, 3), dtype=torch.float32, device=self.device)
        t_translation_batch = torch.zeros((self.last_images_applied, 3), dtype=torch.float32, device=self.device)
        # initial pose parameters from previous optimizations!

        target_images_list_rgb = []
        target_images_list_grey = []
        target_image_mask_list = []
        matching_mask_list = []
        if self.task["texture use world map node points for masking chamfer pcs"]:
            matches_with_world_map_list = []
            points_2D_map_list = []
            split_indices_crossing_list = []
            split_indices_border_list = []

        for i, graph_images in enumerate(self.seen_images[: self.last_images_applied]):
            graph_image = graph_images[2]
            r_rotation_batch[i, :] = graph_image.r_rotation
            t_translation_batch[i, :] = graph_image.t_translation
            if target_images is None:
                _, image_rgb, image_grey = self.UpComingDataGenerator.load_and_create_graph_image_object(
                    desired_batch_nr=graph_images[0],
                    item_in_batch=graph_images[1],)

                if self.texture_type == TextureTypes.VERTEX_TEXTURE_RGB or self.texture_type == TextureTypes.VERTEX_TEXTURE:
                    target_images_list_rgb.append(image_rgb)
                if self.texture_type == TextureTypes.VERTEX_TEXTURE_GREY:
                    target_images_list_grey.append(image_grey)
                target_image_mask_list.append(graph_image.generate_mask_for_image(return_as_torch=True))
            if self.pose_estimation_only:
                matching_mask_list.append(graph_image.generate_matching_mask_for_image(return_as_torch=True))
                if PLOT_MASKED_IMAGE_FOR_TESTURE_USAGE:
                    if self.texture_type == TextureTypes.VERTEX_TEXTURE_RGB or self.texture_type == TextureTypes.VERTEX_TEXTURE:
                        plot_image_with_mask(image_rgb, graph_image.generate_matching_mask_for_image(return_as_torch=True))
                    if self.texture_type == TextureTypes.VERTEX_TEXTURE_GREY:
                        plot_image_with_mask(image_grey, graph_image.generate_matching_mask_for_image(return_as_torch=True), is_grey=True)

                if self.task["texture use world map node points for masking chamfer pcs"]:
                    matches_with_world_map_list.append(graph_image.get_matches_to_world_map)
                    points_2D_map_list.append(torch.from_numpy(graph_image.get_all_image_nodes).to(device=self.device))
                    split_indices_crossing_list.append(graph_image.get_crossing_node_indices_by_removed_border)
                    split_indices_border_list.append(graph_image.get_end_node_indices_by_removed_border)


        if target_images is None:
            if self.texture_type == TextureTypes.VERTEX_TEXTURE_RGB or self.texture_type == TextureTypes.VERTEX_TEXTURE:
                self.target_images = torch.tensor(target_images_list_rgb, device=self.device)/255
            if self.texture_type == TextureTypes.VERTEX_TEXTURE_GREY:
                target_image_grey = (torch.tensor(target_images_list_grey, device=self.device)/255)
                self.target_images = torch.stack((target_image_grey, target_image_grey, target_image_grey), dim=3) #target_image_grey in rgb channel!
        else:
            if self.texture_type == TextureTypes.VERTEX_TEXTURE_GREY:
                self.target_images = torch.stack((target_images, target_images, target_images),
                                                 dim=3)  # target_image_grey in rgb channel!
            else:
                self.target_images = target_images

        if self.image_mask is None:
            self.image_mask = torch.vstack(target_image_mask_list)

        if self.pose_estimation_only:
            self.matching_image_mask_list = torch.vstack(matching_mask_list)
        else:
            self.matching_image_mask_list = None

        if self.task["texture use world map node points for masking chamfer pcs"] and self.task["use chamfer loss"]:
            self.node_point_container = NodePointContainer(world_point_cloud_map=world_point_cloud_map,
                                                           matches_with_world_map_list=matches_with_world_map_list, points_2D_map_list=points_2D_map_list, split_indices_crossing_list=split_indices_crossing_list, split_indices_border_list=split_indices_border_list)

        return r_rotation_batch, t_translation_batch

    def _update_cam_perspectives(self):
        def slice_batch_to_camera_lists(r_rotation_batch, t_translation_batch):
            target_camera_for_optimization = []
            for i in range(0, r_rotation_batch.shape[0], self.task["batch size for rendering"]):

                    rotation_batch_sliced = r_rotation_batch[i: min(i + self.task["batch size for rendering"], r_rotation_batch.shape[0])]
                    t_translation_batch_sliced = t_translation_batch[i: min(i + self.task["batch size for rendering"], r_rotation_batch.shape[0])]
                    R_batch = self.cam_rodrigues_object.get_rotation_matrix(rotation_batch_sliced)
                    T_batch = self.cam_rodrigues_object.get_translation_matrix(t_translation_batch_sliced)
                    target_camera_for_optimization.append(PerspectiveCameras(device=self.device, R=R_batch,
                                         T=T_batch, focal_length=self.focal_length))
            return target_camera_for_optimization

        if self.target_camera_perspectives is None:
            if self.r_rotation_batch_fixed is not None and self.r_rotation_batch_param is None:
                target_camera_list_for_optimization_fixed = slice_batch_to_camera_lists(r_rotation_batch=self.r_rotation_batch_fixed, t_translation_batch=self.t_translation_batch_fixed)
                self.target_camera_list_for_optimization = target_camera_list_for_optimization_fixed
            elif self.r_rotation_batch_fixed is None and self.r_rotation_batch_param is not None:
                target_camera_list_for_optimization_param = slice_batch_to_camera_lists(r_rotation_batch=self.r_rotation_batch_param, t_translation_batch=self.t_translation_batch_param)
                self.target_camera_list_for_optimization = target_camera_list_for_optimization_param
            elif self.r_rotation_batch_fixed is not None and self.r_rotation_batch_param is not None:
                target_camera_list_for_optimization_param = slice_batch_to_camera_lists(r_rotation_batch=self.r_rotation_batch_param, t_translation_batch=self.t_translation_batch_param)
                target_camera_list_for_optimization_fixed = slice_batch_to_camera_lists(r_rotation_batch = self.r_rotation_batch_fixed, t_translation_batch = self.t_translation_batch_fixed)
                # consider the order! --> latest image is the first image i all batches (since this is the general agreement and also respected in seen_images)
                self.target_camera_list_for_optimization =  target_camera_list_for_optimization_param + target_camera_list_for_optimization_fixed
            else:
                print('We have a problem, no pose variables/parameters were set!')
        else:
            if isinstance(self.target_camera_perspectives, list):
                self.target_camera_list_for_optimization = self.target_camera_perspectives
            else:
                self.target_camera_list_for_optimization = [self.target_camera_perspectives]

    def forward(self, iteration):
        self._update_cam_perspectives()
        loss = {k: torch.tensor(0.0, device=self.device) for k in self.losses}

        size_all_views = 0
        textured_mesh = self.mesh_texture.clone()

        def get_point_cloud(img_torch, mask_torch, threshold=0.45):
            mask = torch.bitwise_and(torch.gt(img_torch[0, :, :, 0], threshold), mask_torch[0])
            idx_matrix = self.idx_matrix[mask]
            # idx_matrix = ((img_torch[0, :, :, 0] > 0.2) & (mask_torch[0] == True)).nonzero().squeeze()
            return idx_matrix

        idx_start = 0
        for j in range(len(self.target_camera_list_for_optimization)):

            images_rendered = self.renderer_texture_observe(textured_mesh, cameras=self.target_camera_list_for_optimization[j].clone()) #lights=self.lights #ToDo render a batch >1 at once!
            # Squared L2 distance between the predicted RGB image and the target
            # image from our dataset
            len_current_batch = images_rendered.shape[0]
            images_predicted = images_rendered[..., :3]

            if self.pose_estimation_only:
                if self.task["use chamfer loss with point cloud extraction over every iteration"] and not self.task['Pose optimization based on chamfer loss using 3D projection!']:
                    # be carefull has some robustness issues an is very sensitive to the rendering setting (espacially to the inverse rendering settings)
                    # prefer "use chamfer loss with single point cloud extraction" mode!
                    if iteration == 0:
                        with torch.no_grad():
                            if self.task["texture use world map node points for masking chamfer pcs"]:

                                projection_matrix= _get_projection_matrix(
                                    camera=self.target_camera_list_for_optimization[j],
                                    image_size_torch_batch=self.image_size_torch_tensor[None])
                                match_idcs_with_world_map = self.node_point_container.get_matches_with_world_map_list[j][:, 0]
                                key_node_points_world_map_3d = self.node_point_container.get_node_points_on_mesh[match_idcs_with_world_map]
                                re_projected_key_node_points_world_map_2d = project_3d_points_to_image_plane_without_distortion(
                                    projection_matrix[0], key_node_points_world_map_3d,
                                    image_size=self.image_size_torch_tensor)
                                #change 2d as done after
                                self.mask_key_matches_predicted_image = generate_mask(node_pos=re_projected_key_node_points_world_map_2d.cpu().numpy().astype(int),
                                    mask_size=(self.target_images.shape[1], self.target_images.shape[1]),
                                    return_as_torch=True, device=self.device)
                            else:
                                self.mask_key_matches_predicted_image = torch.ones(1, self.target_images.shape[1], self.target_images.shape[1]).to(dtype=torch.bool, device=self.device)

                    #with torch.no_grad():
                    self.y_img_target_pix_pos_of_patterns = get_point_cloud(
                        img_torch=self.target_images[idx_start: idx_start + len_current_batch],
                        mask_torch=self.matching_image_mask_list[idx_start: idx_start + len_current_batch],
                        threshold=0.55)
                    y_img_target_pix_pos_of_patterns = self.y_img_target_pix_pos_of_patterns
                    x_point_cloud_predicted = get_point_cloud(img_torch=images_predicted, mask_torch=self.mask_key_matches_predicted_image[
                                                                       idx_start: idx_start + len_current_batch], threshold=0.50)

                    img_predicted_pix_pos_of_patterns = x_point_cloud_predicted.type(dtype=torch.long)
                    # img_predicted_pix_pos_of_patterns_in = torch.stack((img_predicted_pix_pos_of_patterns[:, 1], img_predicted_pix_pos_of_patterns[:,0])).t()
                    # img_predicted_pix_pos_of_patterns_in = img_predicted_pix_pos_of_patterns
                    img_predicted_pix_pos_of_patterns[:, 1] = img_predicted_pix_pos_of_patterns[:, 1] - 1  #ToDo: compare with deformation!
                    texture_points_3d, _ = self.inverse_renderer(meshes_world=self.mesh_texture,
                                                                                  node_pos= img_predicted_pix_pos_of_patterns, cameras= self.target_camera_list_for_optimization[j],
                                                                                   req_gradient=True, req_fragments=False)
                    projected_image_keypoints_2d = camera_projection(texture_points_3d.points_packed(), camera=self.target_camera_list_for_optimization[j],
                                                                     image_size=self.image_size_torch_tensor, device=self.device, r_rotation=None, t_translation=None)

                    error, _ = chamfer_distance(x=projected_image_keypoints_2d[None], y=y_img_target_pix_pos_of_patterns[None])
                    loss["chamfer every iteration loss"] += error * len_current_batch

                    #print('Chamfer loss error: ', error)

                if self.task["use chamfer loss with single point cloud extraction"] and not self.task['Pose optimization based on chamfer loss using 3D projection!']:
                    def get_point_cloud(img_torch, mask_torch, threshold=0.45):
                        mask = torch.bitwise_and(torch.gt(img_torch[0, :, :, 0], threshold), mask_torch[0])
                        idx_matrix = self.idx_matrix[mask]
                        # idx_matrix = ((img_torch[0, :, :, 0] > 0.2) & (mask_torch[0] == True)).nonzero().squeeze()
                        return idx_matrix

                    if iteration == 0:
                        with torch.no_grad():
                            if self.task["texture use world map node points for masking chamfer pcs"]:
                                projection_matrix = _get_projection_matrix(
                                    camera=self.target_camera_list_for_optimization[j],
                                    image_size_torch_batch=self.image_size_torch_tensor[None])
                                match_idcs_with_world_map = \
                                self.node_point_container.get_matches_with_world_map_list[j][:, 0]
                                key_node_points_world_map_3d = self.node_point_container.get_node_points_on_mesh[
                                    match_idcs_with_world_map]
                                re_projected_key_node_points_world_map_2d = project_3d_points_to_image_plane_without_distortion(
                                    projection_matrix[0], key_node_points_world_map_3d,
                                    image_size=self.image_size_torch_tensor)
                                # change 2d as done after
                                self.mask_key_matches_predicted_image = generate_mask(
                                    node_pos=re_projected_key_node_points_world_map_2d.cpu().numpy().astype(int),
                                    mask_size=(self.target_images.shape[1], self.target_images.shape[1]),
                                    return_as_torch=True, device=self.device)

                            else:
                                self.mask_key_matches_predicted_image = torch.ones(1, self.target_images.shape[1],
                                                                                   self.target_images.shape[1]).to(dtype=torch.bool, device=self.device)

                                # self.mask_key_matches_predicted_image = self.matching_image_mask_list[
                                #            idx_start: idx_start + len_current_batch] #problematic since it doesnt move


                            # get 2d point cloud of target image
                            self.y_img_target_pix_pos_of_patterns = get_point_cloud(
                                img_torch=self.target_images[idx_start: idx_start + len_current_batch],
                                mask_torch=self.matching_image_mask_list[
                                           idx_start: idx_start + len_current_batch],
                                threshold=0.55)
                            # get 3d points on mesh
                            x_point_cloud_predicted = get_point_cloud(img_torch=images_predicted,
                                                                      mask_torch=self.mask_key_matches_predicted_image[
                                                                                 idx_start: idx_start + len_current_batch],
                                                                      threshold=0.50)

                            img_predicted_pix_pos_of_patterns = x_point_cloud_predicted.type(dtype=torch.long)
                            img_predicted_pix_pos_of_patterns[:, 1] = img_predicted_pix_pos_of_patterns[:, 1] -1 #ToDo: only necessary for the predicted image not for the target image point cloud extraction! - can be proofed by plotting all variations!
                            texture_points_3d, _ = self.inverse_renderer(meshes_world=self.mesh_texture,
                                                                         node_pos=img_predicted_pix_pos_of_patterns,
                                                                         cameras=
                                                                         self.target_camera_list_for_optimization[
                                                                             j],
                                                                         req_gradient=True, req_fragments=False)
                            self.point_cloud_on_mesh_3d = texture_points_3d.points_packed()

                    projected_image_keypoints_2d = camera_projection(self.point_cloud_on_mesh_3d , camera=self.target_camera_list_for_optimization[j],
                                                                     image_size=self.image_size_torch_tensor,
                                                                     device=self.device, r_rotation=None,
                                                                     t_translation=None)

                    y_img_target_pix_pos_of_patterns = self.y_img_target_pix_pos_of_patterns

                    error, _ = chamfer_distance(x=projected_image_keypoints_2d[None],
                                                y=y_img_target_pix_pos_of_patterns[None])
                    loss["chamfer single iteration loss"] += error * len_current_batch
                    #print('Chamfer loss error: ', error)
                    #_____________________________________________________________________________________________________

                if self.task["Pose optimization based on chamfer loss using 3D projection!"]:
                    def get_point_cloud(img_torch, mask_torch=None, threshold=0.45):
                        if mask_torch is None:
                            mask_torch = torch.ones(1, self.target_images.shape[1],
                                       self.target_images.shape[1]).to(dtype=torch.bool, device=self.device)
                        mask = torch.bitwise_and(torch.gt(img_torch[0, :, :, 0], threshold), mask_torch[0])
                        idx_matrix = self.idx_matrix[mask]
                        # idx_matrix = ((img_torch[0, :, :, 0] > 0.2) & (mask_torch[0] == True)).nonzero().squeeze()
                        return idx_matrix

                    if iteration == 0:
                        with torch.no_grad():

                            projection_matrix = _get_projection_matrix(
                                camera=self.target_camera_list_for_optimization[j],
                                image_size_torch_batch=self.image_size_torch_tensor[None])

                            mesh_grey_features = self.mesh_texture.textures.verts_features_packed()
                            #feature_threshold_for_robustness = torch.nn.Threshold(threshold=0.6, value=0.0)
                            mesh_grey_features_thresholded = self.feature_threshold_for_robustness(mesh_grey_features)

                            torch.ones(mesh_grey_features.shape).to(device=self.device)
                            idx_set_features = torch.argwhere(mesh_grey_features_thresholded)

                            self.global_point_patterns_form_textured_mesh = self.mesh_texture.verts_packed()[idx_set_features[:,  0]]
                            self.global_normal_patterns_form_textured_mesh = self.mesh_texture.verts_normals_packed()[idx_set_features[:, 0]]

                        # get 2d point cloud of target image
                            # Processing observed image data
                            self.y_img_target_pix_pos_of_patterns = get_point_cloud(
                                img_torch=self.target_images[idx_start: idx_start + len_current_batch],
                                mask_torch=self.image_mask[idx_start: idx_start + len_current_batch], threshold=0.55)
                            # get 3d points on mesh
                            self.img_observed_pix_pos_of_patterns = self.y_img_target_pix_pos_of_patterns.type(dtype=torch.long)
                            self.img_observed_pix_pos_of_patterns [:, 1] = self.img_observed_pix_pos_of_patterns [:,1] - 1  # ToDo: only necessary for the predicted image not for the target image point cloud extraction! - can be proofed by plotting all variations!

                    texture_points_3d_from_observed_target, _ = self.inverse_renderer(meshes_world=self.mesh_texture, #Settings for Chamfer reconstruction!!
                                                                     node_pos=self.img_observed_pix_pos_of_patterns ,
                                                                     cameras=
                                                                     self.target_camera_list_for_optimization[j],
                                                                     req_gradient=True, req_fragments=False)

                    #____________________
                    # #To Delete jus for experimenting
                    # self.mesh_texture.textures = TexturesVertex(verts_features=sphere_verts_rgb)
                    #
                    # idx_faces = fragment.pix_to_face[0, 0]
                    #
                    # _, idx_vertices = self.mesh_texture.get_mesh_verts_faces(0)
                    # idx_vertices = self.mesh_texture.faces_packed()
                    #
                    # self.mesh_texture.get_mesh_verts_faces(faces)[faces]
                    #
                    #
                    #______________________

                    if self.task["Pose optimization based on chamfer loss using 3D projection - Plot point cloud on mesh!"] is True:
                        if iteration % self.task["Pose optimization based on chamfer loss using 3D projection - Plot point cloud on mesh ever-ith iteration!"] == 0:
                            fig_pose_optimization_chamfer_3D_on_mesh, ax = plot_point_cloud_on_mesh(geometry_mesh=self.geometry_mesh, point_cloud_observed = texture_points_3d_from_observed_target.points_packed(),
                                                               point_cloud_target_mesh = self.global_point_patterns_form_textured_mesh, fig=None, ax=None, camera=None)
                            fig_pose_optimization_chamfer_3D_on_mesh.show()
                            save_figure(fig=fig_pose_optimization_chamfer_3D_on_mesh, ax = ax,
                                        name_of_figure='fig_pose_optimization_chamfer_3D_on_mesh_' + str(iteration) + '_iteration_'+str(self.ith_call)+'_nr_of_call',
                                        show_fig=False)
                        # point_cloud_target_mesh = self.global_point_patterns_form_textured_mesh
                        # point_cloud_observed = texture_points_3d_from_observed_target.points_packed()
                    error, _ = chamfer_distance(x= self.global_point_patterns_form_textured_mesh[None], y=texture_points_3d_from_observed_target.points_packed()[None] ,
                                                x_normals=self.global_normal_patterns_form_textured_mesh[None], y_normals=texture_points_3d_from_observed_target.normals_packed()[None])
                    # error, _ = chamfer_distance(x=self.global_point_patterns_form_textured_mesh[None],
                    #                             y=texture_points_3d_from_observed_target.points_packed()[None]
                    #                             )
                    #print('Chamfer 3D loss: ', error, 'print iteration ', iteration)
                    loss["chamfer loss 3d data"] += error * len_current_batch * 600


                if not self.task["Pose optimization based on chamfer loss using 3D projection!"] and iteration % self.task["use chamfer loss - plot period for every-ith iteration!"] == 0:
                    if self.task["use chamfer loss - make plots"] and (self.task["use chamfer loss with single point cloud extraction"] or self.task["use chamfer loss with point cloud extraction over every iteration"]):

                        # Remember, the projected_image_keypoints_2d don't align up perfectly with the image,
                        # since, the re-projection, blurring, floating vs. int declaration causes a mismatch
                        fig_chamfer_pcs_pose_recon, ax_chamfer_pcs_pose_recon = visualize_features_on_image(image=images_predicted,
                                                    pos1=projected_image_keypoints_2d, pos2=y_img_target_pix_pos_of_patterns, node_thickness=20, title='Features on Texture Model with Extracted and Target Features')

                        # pos1 = projected_image_keypoints_2d, pos2 = y_img_target_pix_pos_of_patterns (as it is for deformation reconstruction)
                        self.fig_ax_chamfer_loss_pcs_pose_recon = (fig_chamfer_pcs_pose_recon, ax_chamfer_pcs_pose_recon)

                    if self.task["use chamfer loss - make plots - all variations"] and iteration % self.task["use chamfer loss - make plots - all variations in period of"] == 0:

                        fig_chamfer_pcs_pose_recon, ax_chamfer_pcs_pose_recon = visualize_features_on_image(image=images_predicted,
                                                pos1=projected_image_keypoints_2d, pos2=y_img_target_pix_pos_of_patterns, node_thickness=20, title='Features on Texture Model with Extracted and Target Features and iteration nr '+str(iteration))
                        fig_chamfer_pcs_pose_recon.show()

                        y_img_target_pix_pos_of_patterns   = y_img_target_pix_pos_of_patterns .type(dtype=torch.long)
                        # img_predicted_pix_pos_of_patterns[:, 1] = img_predicted_pix_pos_of_patterns[:, 1] - 1#ToDo: only necessary for the predicted image not for the target image point cloud extraction! - can be proofed by plotting all variations!
                        y_img_target_pix_pos_of_patterns[:, 1] =y_img_target_pix_pos_of_patterns[:,
                                                                  1] - 1  # ToDo: only necessary for the predicted image not for the target image point cloud extraction! - can be proofed by plotting all variations!
                        texture_points_3d, _ = self.inverse_renderer(meshes_world=self.mesh_texture,
                                                                     node_pos=y_img_target_pix_pos_of_patterns,
                                                                     cameras=
                                                                     self.target_camera_list_for_optimization[
                                                                         j],
                                                                     req_gradient=True, req_fragments=False)
                        y_point_cloud_on_mesh_3d = texture_points_3d.points_packed()
                        y_projected_image_keypoints_2d = camera_projection(y_point_cloud_on_mesh_3d, camera=
                        self.target_camera_list_for_optimization[j],
                                                                         image_size=self.image_size_torch_tensor,
                                                                         device=self.device, r_rotation=None,
                                                                         t_translation=None)
                        fig, ax = visualize_features_on_image(image=images_predicted,
                                                              pos1=projected_image_keypoints_2d,
                                                    pos2=y_projected_image_keypoints_2d, node_thickness=20,
                                                    title='features on texture model with reprojected target image features and iteration nr '+str(iteration))
                        fig.show()
                        fig, ax = visualize_features_on_image(image=self.target_images[idx_start: idx_start + len_current_batch],
                                                             pos1 = projected_image_keypoints_2d,
                                                    pos2=y_projected_image_keypoints_2d , node_thickness=20,
                                                    title='features of texture model on target image and with target image features and iteration nr '+str(iteration))
                        fig.show()

                # if self.task["Set features on mesh texture for a skeletonized image by inverse rendering function!"]:
                #     def get_point_cloud(img_torch, mask_torch=None, threshold=0.45):
                #         if mask_torch is None:
                #             mask_torch = torch.ones(1, self.target_images.shape[1],
                #                        self.target_images.shape[1]).to(dtype=torch.bool, device=self.device)
                #         mask = torch.bitwise_and(torch.gt(img_torch[0, :, :, 0], threshold), mask_torch[0])
                #         idx_matrix = self.idx_matrix[mask]
                #         # idx_matrix = ((img_torch[0, :, :, 0] > 0.2) & (mask_torch[0] == True)).nonzero().squeeze()
                #         return idx_matrix
                #
                #     if iteration == 0:
                #         with torch.no_grad():
                #
                #             correct_for_close_overlapping_dismatch = False
                #             if correct_for_close_overlapping_dismatch:
                #                 #ToDo: To be implemented!
                #                 mesh_grey_features = self.mesh_texture.textures.verts_features_packed()
                #                 #feature_threshold_for_robustness = torch.nn.Threshold(threshold=0.6, value=0.0)
                #                 mesh_grey_features_thresholded = self.feature_threshold_for_robustness(mesh_grey_features)
                #
                #                 torch.ones(mesh_grey_features.shape).to(device=self.device)
                #                 idx_set_features = torch.argwhere(mesh_grey_features_thresholded)
                #
                #                 self.global_point_patterns_form_textured_mesh = self.mesh_texture.verts_packed()[idx_set_features[:,  0]]
                #                 self.global_normal_patterns_form_textured_mesh = self.mesh_texture.verts_normals_packed()[idx_set_features[:, 0]]
                #
                #         # get 2d point cloud of target image
                #             # Processing observed image data
                #             self.y_img_target_pix_pos_of_patterns = get_point_cloud(
                #                 img_torch=self.target_images[idx_start: idx_start + len_current_batch],
                #                 mask_torch=self.image_mask[idx_start: idx_start + len_current_batch], threshold=0.55)
                #             # get 3d points on mesh
                #             self.img_observed_pix_pos_of_patterns = self.y_img_target_pix_pos_of_patterns.type(dtype=torch.long)
                #             self.img_observed_pix_pos_of_patterns [:, 1] = self.img_observed_pix_pos_of_patterns [:,1] - 1  # ToDo: only necessary for the predicted image not for the target image point cloud extraction! - can be proofed by plotting all variations!
                #
                #     texture_points_3d_from_observed_target, _ = self.inverse_renderer(meshes_world=self.mesh_texture, #Settings for Chamfer reconstruction!!
                #                                                      node_pos=self.img_observed_pix_pos_of_patterns ,
                #                                                      cameras=
                #                                                      self.target_camera_list_for_optimization[j],
                #                                                      req_gradient=True, req_fragments=False)

                    if self.task["Pose optimization based on chamfer loss using 3D projection - Plot point cloud on mesh!"]:
                        if iteration % self.task["Pose optimization based on chamfer loss using 3D projection - Plot point cloud on mesh ever-ith iteration!"] == 0:
                            fig_pose_optimization_chamfer_3D_on_mesh, ax = plot_point_cloud_on_mesh(geometry_mesh=self.geometry_mesh, point_cloud_observed = texture_points_3d_from_observed_target.points_packed(),
                                                               point_cloud_target_mesh = self.global_point_patterns_form_textured_mesh, fig=None, ax=None, camera=None)
                            fig_pose_optimization_chamfer_3D_on_mesh.show()
                            save_figure(fig=fig_pose_optimization_chamfer_3D_on_mesh, ax=ax,
                                        name_of_figure='fig_pose_optimization_chamfer_3D_on_mesh_' + str(iteration) + '_iteration_'+str(self.ith_call)+'_nr_of_call',
                                        show_fig=False)
                        # point_cloud_target_mesh = self.global_point_patterns_form_textured_mesh
                        # point_cloud_observed = texture_points_3d_from_observed_target.points_packed()
                        error, _ = chamfer_distance(x= self.global_point_patterns_form_textured_mesh[None], y=texture_points_3d_from_observed_target.points_packed()[None] ,
                                                    x_normals=self.global_normal_patterns_form_textured_mesh[None], y_normals=texture_points_3d_from_observed_target.normals_packed()[None])
                        # error, _ = chamfer_distance(x=self.global_point_patterns_form_textured_mesh[None],
                        #                             y=texture_points_3d_from_observed_target.points_packed()[None]
                        #                             )
                        #print('Chamfer 3D loss: ', error, 'print iteration ', iteration)
                        loss["chamfer loss 3d data"] += error * len_current_batch * 600

                if self.task["use image pattern loss for pose"]:


                    images_rendered_pattern_high_diff = self.renderer_texture_high_diff(textured_mesh, cameras=self.target_camera_list_for_optimization[j].clone())  # lights=self.lights #ToDo render a batch >1 at once!
                    images_predicted_pattern_high_diff = images_rendered_pattern_high_diff[..., :3]
                    # visualize_image(img = images_predicted_pattern_high_diff)
                    if iteration == 0:
                        self.image_grey_scale_shift = torch.full(size=images_predicted_pattern_high_diff.shape, fill_value=self.task[
                            'Texture optimization - shift by an grey value range: float between zero and one'], device=self.device)
                    error_img = images_predicted_pattern_high_diff - self.target_images[idx_start: idx_start + len_current_batch] - self.image_grey_scale_shift  #ToDO: visualize image renderings of using the different renderers! (compare!)
                    error = error_img[self.matching_image_mask_list[idx_start: idx_start + len_current_batch]]
                    loss_error = (error ** 2).mean()

                    loss["image pattern loss"] += loss_error* len_current_batch

                    if self.task["use chamfer loss - make plots - all variations"] and iteration % self.task[
                        "use chamfer loss - make plots - all variations in period of"] == 0:
                        diff_error_img = images_predicted + self.target_images[idx_start: idx_start + len_current_batch]
                        diff_error_img_np = diff_error_img.detach().cpu().numpy()[0]
                        fig, ax = visualize_image(img=diff_error_img_np, size_width=3.75, size_height=3.75, title='Error between predicted and target image, nr of iteration: '+str(iteration))
                        fig.show()
                        fig, ax = visualize_diff_off_images(img1 = images_predicted[0].clone().cpu().detach().numpy(), img2=self.target_images[idx_start: idx_start + len_current_batch][0].clone().cpu().detach().numpy() ,size_width=3.75, size_height=3.75, title='Error between predicted and target image, nr of iteration: '+str(iteration))
                        fig.show()
                        # fig, ax = visualize_image(img=images_predicted.detach().cpu().numpy()[0], size_width=3.75, size_height=3.75, title='Error image between predicted and target image and nr of iteration '+str(iteration))
                        # fig.show()
                        # fig, ax = visualize_image(img=self.target_images[idx_start: idx_start + len_current_batch].detach().cpu().numpy()[0], size_width=3.75, size_height=3.75, title='Error image between predicted and target image and nr of iteration '+str(iteration))
                        # fig.show()
            else:
                if iteration == 0:
                    self.image_grey_scale_shift = torch.full(size=images_predicted.shape, fill_value=self.task[
                        'Texture optimization - shift by an grey value range: float between zero and one'], device=self.device)
                error_img = images_predicted - self.target_images[idx_start: idx_start + len_current_batch] - self.image_grey_scale_shift
                error_img =  error_img[self.image_mask[idx_start: idx_start + len_current_batch]]
                loss_error = (error_img ** 2).mean()
                #print('Error Texture Reconstruction: ', loss_error)
                loss["texture reconstruction loss"] += loss_error * len_current_batch

                if iteration % 20 == 0:
                    diff_error_img = images_predicted + self.target_images[idx_start: idx_start + len_current_batch] #ToDo show difference in different colors!
                    diff_error_img_np = diff_error_img.detach().cpu().numpy()[0]
                    fig, ax = visualize_image(img=diff_error_img_np, size_width=3.75, size_height=3.75,
                                              title='Error between predicted and target image, nr of iteration: ' + str(iteration))
                    fig.show()
                    # fig, ax = visualize_diff_off_images(img1=images_predicted[0].clone().cpu().detach().numpy(), img2=
                    # self.target_images[idx_start: idx_start + len_current_batch][0].clone().cpu().detach().numpy(),
                    #                                     size_width=3.75, size_height=3.75,
                    #                                     title='Error between predicted and target image, nr of iteration: ' + str(
                    #                                         iteration))
                    # fig.show()

            size_all_views += len_current_batch
            idx_start = idx_start + len_current_batch


        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=self.device)
        for k, l in loss.items():
            sum_loss += (l * self.losses[k]["weight"])/size_all_views
            self.losses[k]["values"].append(float(l.detach().cpu())*self.losses[k]["weight"])

        return sum_loss
















    def update_losses(self, losses):
        self.losses = losses

    def update_task(self, task):
        if task is not None:
            self.task.update(task)

    def visualize_texture_prediction(self, title:str= 'texture fitting vs. target image'):
        visualize_texture_prediction(textured_mesh=self.get_textured_mesh(),
                                     renderer=self.renderer_texture_observe,
                                     camera_perspective=self.target_camera_list_for_optimization[0],  # ToDo vary view that is compared!
                                     image_to_compare_torch=self.target_images[0],
                                     title=title, grey=self.texture_type==TextureTypes.VERTEX_TEXTURE_GREY)
        return
    def visualize_mesh(self, title:str= 'Model with texture'):
        fig, ax = visualize_mesh(textured_mesh=self.get_textured_mesh(), renderer=self.renderer_view,
                       camera_perspective=self.camera_view,
                       title=title, grey=self.texture_type==TextureTypes.VERTEX_TEXTURE_GREY)

    def visualize_mesh_over_batch(self, title: str = 'Model with texture', batch_size=8):
        # Set batch size - this is the number of different viewpoints from which we want to render the mesh.

        renderer_view_perspective = self.renderer_view
        # renderer_view_perspective.shader
        # # Move the light back in front of the cow which is facing the -z direction.
        # # Move the light location so the light is shining on the cow's face.
        renderer_view_perspective.shader.lights.location = torch.tensor([[2.0, 2.0, -2.0]], device=self.device)
        # # Change specular color to green and change material shininess
        renderer_view_perspective.shader.materials = Materials(
            device=self.device,
            specular_color=[[0.8, 0.2, 0.1]],
            shininess=10.0
        )
        elev = torch.linspace(0, 180, batch_size)
        azim = torch.linspace(0, 180, batch_size)
        R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim, device=self.device)
        camera_view_perspective = self.camera_view
        for i in range(batch_size):
            camera_view_perspective.R = R[i][None]
            camera_view_perspective.T = T[i][None]
            renderer_view_perspective.shader.lights.location = camera_view_perspective.get_camera_center()
            fig, ax = visualize_mesh(textured_mesh=self.get_textured_mesh(), renderer=renderer_view_perspective,
                                     camera_perspective=camera_view_perspective,
                                     title=title, grey=self.texture_type == TextureTypes.VERTEX_TEXTURE_GREY)
            fig.show()
            ax.title.set_visible(False)
            save_figure(fig=fig, name_of_figure='mesh_view_perspective_rendering_in_texture_model_' + str(
                elev[i].numpy()) + '_azim_' + str(azim[i].numpy()))

    def get_rotation_translation(self):
        rotation = None
        translation = None
        if self.r_rotation_batch_fixed is not None and self.t_translation_batch_fixed is not None:
            rotation = self.r_rotation_batch_fixed
            translation = self.t_translation_batch_fixed
        if self.r_rotation_batch_param is not None and self.t_translation_batch_param is not None:
            r_rotation_batch_param = self.r_rotation_batch_param.clone().detach()
            t_translation_batch_param = self.t_translation_batch_param.clone().detach()
            r_rotation_batch_param.requires_grad = False
            t_translation_batch_param.requires_grad = False

            if rotation is not  None and translation is not None:
                rotation = torch.vstack((r_rotation_batch_param, rotation))
                translation = torch.vstack((t_translation_batch_param, translation))
            else:
                rotation = r_rotation_batch_param
                translation = t_translation_batch_param
        return rotation, translation

    @property
    def get_idx_of_last_image_applied(self):
        return self.last_images_applied

    def get_textured_mesh(self):
        #return self.mesh_texture

        self.mesh_texture.textures.verts_features_padded().requires_grad = False
        # self.sphere_verts_rgb = nn.Parameter(
        #     self.mesh_texture.textures.verts_features_padded().clone().requires_grad_(True))
        self.mesh_texture.textures = TexturesVertex(verts_features=self.mesh_texture.textures.verts_features_padded().clone().detach())
        return self.mesh_texture

    @property
    def get_loss_dict_history(self):
        return self.losses

    @staticmethod
    def get_index_matrix(image_size: int, device='cpu'):
        idx_matrix = torch.zeros((image_size, image_size, 2))
        for i in range(idx_matrix.shape[0]):
            idx_matrix[i, :, 0] = torch.full((1, image_size), i)[0]
            idx_matrix[i, :, 1] = torch.arange(0, image_size)

        return idx_matrix.to(device=device)

    @property
    def get_fig_chamfer_loss_reconstruction(self):
        return self.fig_chamfer_loss_pcs_pose_recon

def _get_projection_matrix(camera: PerspectiveCameras, image_size_torch_batch: torch.Tensor = None):
    if image_size_torch_batch is None:
        image_size_torch_batch = camera.image_size[None]
        #image_size_torch_batch =torch.tensor([128, 128])[None]
    R_cv, t_cv, K_cv = opencv_from_cameras_projection(cameras=camera, image_size=image_size_torch_batch) # ToDo check if K_cv makes sense (shift of principle points)
    torch_cam_3d_batch = [CameraTorch(R_cv=R_cv[i], t_cv=t_cv[i][None], K=K_cv[i],
                                       image_size=image_size_torch_batch, device=R_cv.device) for i in range(R_cv.shape[0])]
    proj_matrix_batch = [torch_cam_3d_batch[i].projection for i in range(R_cv.shape[0])]
    # This camera is using the cv literature convention --> R,T,K needs to be converted!
    return torch.stack(proj_matrix_batch)

# def get_point_cloud_tmp(img_torch, mask_torch, threshold=0.45):
#     idx = ((img_torch[0, :, :, 0] > 0.5) & (mask_torch[0] == True)).nonzero().squeeze()
#     return idx.float()[None]