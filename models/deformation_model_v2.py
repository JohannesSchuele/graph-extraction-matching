import matplotlib.pyplot as plt
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
from configs.plot.colours import *
from models.camera_pose import camera_projection, plot_img_2d_points
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
from utils.plot_images import visualize_image
from utils.mesh_operations import subdivideMesh, pool_mesh_to_dim_of_original_mesh
from configs.config import *
from utils.deformation_helpers import NodePointContainer
from utils.deformation_helpers import get_index_matrix, get_point_cloud
from tools_generate.mask import generate_mask
from tools_generate.image_processing import skeletonise_and_clean
from skimage.filters import threshold_otsu
class DeformationModelTexture(nn.Module):
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
    def __init__(self, geometry_mesh, texture, texture_type, seen_images, RenderersCollection, UpComingDataGenerator=None,  data_rotation_translation_batch_lists=None,
                       data_target_imag_batch:torch.Tensor=None, data_mask_batch_lists=None, world_point_cloud_map=None,  node_point_container_extern=None,
                 inverse_renderer=None, camera_view=None, ith_call=0,
                 loaded_target_cameras=None, losses=None, task=None):
        super().__init__()
        self.seen_images = seen_images
        self.renderer_texture_observe = RenderersCollection.renderer_texture
        self.renderer_texture_high_diff = RenderersCollection.renderer_texture_high_diff
        self.focal_length = RenderersCollection.focal_length
        self.renderer_view = RenderersCollection.renderer_view
        self.inverse_renderer = RenderersCollection.inverse_renderer
        self.lights_view_point_lights = RenderersCollection.lights_view_point_lights
        self.lights_ambient = RenderersCollection.lights_ambient

        self.image_size_torch_tensor = torch.tensor([RenderersCollection.image_size, RenderersCollection.image_size], device=RenderersCollection.device)
        # self.renderer_texture = renderer_texture
        # self.focal_length = renderer_texture.rasterizer.cameras.focal_length
        # self.renderer_view = renderer_view
        self.camera_view = camera_view
        self.device = geometry_mesh.device
        self.geometry_mesh = geometry_mesh
        self.texture = texture
        self.ith_call = ith_call

        if losses is not None:
            self.losses = losses
        else:
            self.losses = {"texture": {"weight": 0.6, "values": []}}

        self.task = {
                     "texture based chamfer optimization": False,
                     "deformation use texture comparison":False,
                     "number of last images 2b used for deformation": 1,
                     "deformation use chamfer loss": True,
                     "deformation laod data from seen_image list": True,
                     "deformation use node point landmarks on world map": False,
                     "deformation use world map node points for masking chamfer pcs": True,
                     'deformation - visualization of chamfer loss every iteration for gif building': False,
                     }
        if task is not None:
            self.task.update(task)
        if self.task["deformation use chamfer loss"]:
            self.idx_matrix = get_index_matrix(device=self.device, image_size_int=128)

        if self.task['deformation - visualization of chamfer loss every iteration for gif building']: #self.task['deformation - visualization of chamfer loss and texture gifs']:
            self.Gif_deformation_chamfer_loss_texture = GifMaker(name_of_gif='Chamfer_loss_deformation_reconstruction_texture_ith_call' + str(self.ith_call))
            self.Gif_deformation_chamfer_loss_view = GifMaker(name_of_gif='Chamfer_loss_deformation_reconstruction_view_ith_call' + str(self.ith_call))

        self.last_images_2bused = self.task["number of last images 2b used for deformation"] # last_images_2bused must be at least >=2 if only camera pose it to be optimized! -> else there are no parameters!
            #just one at the sime time
        # self.pose_estimation_only = self.task["pose by texture"] and self.task["texture learning"] is False
        # self.texture_learning_only = self.task["texture learning"] and self.task["pose by texture"] is False
        # self.texture_and_pose_learning = self.task["texture learning"] and self.task["pose by texture"] is True
        self.fig_ax_chamfer_loss_pcs_pose_recon = None
        self.UpComingDataGenerator = UpComingDataGenerator

        if texture_type is not None:
            self.texture_type = texture_type
        else:
            self.texture_type = TextureTypes.VERTEX_TEXTURE_RGB


        if self.seen_images is not None and self.task["deformation laod data from seen_image list"]:
            r_rotation_batch, t_translation_batch = self._get_poses_target_images(world_point_cloud_map=world_point_cloud_map)
        else:
            r_rotation_batch = data_rotation_translation_batch_lists[0]
            t_translation_batch = data_rotation_translation_batch_lists[1] #ToDo add masks etc!!

            self.image_mask = data_mask_batch_lists[0]
            self.matching_image_mask_list = data_mask_batch_lists[0]

            if data_target_imag_batch is not None:
                if self.texture_type == TextureTypes.VERTEX_TEXTURE_GREY:
                    target_image_grey = data_target_imag_batch
                    self.target_images = torch.stack((target_image_grey, target_image_grey, target_image_grey), dim=3)
                else:
                    self.target_images = data_target_imag_batch
            if self.task["deformation use node point landmarks on world map"] or self.task["deformation use world map node points for masking chamfer pcs"]:
                self.node_point_container = node_point_container_extern

        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=self.device)
        self.target_camera_perspective_list = self.slice_batch_to_camera_lists(r_rotation_batch, t_translation_batch)
        if self.task["deformation use node point landmarks on world map"] or self.task["deformation use world map node points for masking chamfer pcs"]:
            self.projection_matrices = torch.vstack([_get_projection_matrix(camera=self.target_camera_perspective_list[i], image_size_torch_batch=self.image_size_torch_tensor[None]) for i in
                 range(self.target_camera_perspective_list.__len__())])


        # if self.texture_learning_only:
        #     self._define_texture_parameters(mesh=mesh, texture_type=texture_type)
        #     self._parametrize_pose_batches_2lists(r_rotation_batch, t_translation_batch)
        #     # self.r_rotation_batch = r_rotation_batch
        #     # self.t_translation_batch = t_translation_batch
        # elif self.pose_estimation_only:
        #     self._parametrize_pose_batches_2lists(r_rotation_batch, t_translation_batch)
        #     #self._define_texture(mesh=mesh, texture_type=texture_type)
        #     self.mesh_texture = mesh
        #
        # else: # both is optimized at the same time!
        #     self._define_texture_parameters(mesh=mesh, texture_type=texture_type)
        #     self._parametrize_pose_batches_2lists(r_rotation_batch, t_translation_batch)

        # self.param_r_rotation_batch = None
        # self.param_t_translation_batch = None
        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=self.device)
        # self.target_camera_perspectives = loaded_target_cameras
        self.fig_chamfer_loss_pcs_pose_recon = None
        self.__define_parameters()



    def __define_parameters(self):
        # deform the mesh by offsetting its vertices
        # The shape of the deform parameters is equal to the total number of vertices of the mesh
        self.verts_shape = self.geometry_mesh.verts_packed().shape
        deform_verts = torch.full(self.verts_shape, 0.0, device=self.device, requires_grad=True)
        self.deform_verts_param = nn.Parameter(deform_verts)

    # def _define_texture_parameters(self, mesh, texture_type:TextureTypes):
    #     # self.mesh_texture = mesh.clone()
    #     # del mesh
    #     self.mesh_texture = mesh
    #     if texture_type is TextureTypes.ATLAS_TEXTURE:
    #         # parameter must be stored/ (kept as an attribute) in the nn.Module class!
    #         self.rgb_atlas_param = nn.Parameter(self.mesh_texture.textures.atlas_padded().clone().requires_grad_(True))
    #         #rgb_atlas_param = nn.Parameter(self.mesh_texture.textures.atlas_packed().clone().requires_grad_(True))
    #         self.mesh_texture.textures = TexturesAtlas(self.rgb_atlas_param)
    #     elif texture_type is TextureTypes.VERTEX_TEXTURE or texture_type is TextureTypes.VERTEX_TEXTURE_RGB or texture_type is TextureTypes.VERTEX_TEXTURE_GREY:
    #         # learn per vertex colors for our sphere mesh that define texture
    #         self.sphere_verts_rgb = nn.Parameter(self.mesh_texture.textures.verts_features_padded().clone().requires_grad_(True))
    #         self.mesh_texture.textures = TexturesVertex(verts_features=self.sphere_verts_rgb)

    # def _define_texture(self, mesh, texture_type:TextureTypes):
    #     self.mesh_texture = mesh
    #     if texture_type is TextureTypes.ATLAS_TEXTURE:
    #         # parameter must be stored/ (kept as an attribute) in the nn.Module class!
    #         self.rgb_atlas = self.mesh_texture.textures.atlas_padded().clone()
    #         self.mesh_texture.textures = TexturesAtlas(self.rgb_atlas_param)
    #     elif texture_type is TextureTypes.VERTEX_TEXTURE or texture_type is TextureTypes.VERTEX_TEXTURE_RGB:
    #         self.sphere_verts_rgb = self.mesh_texture.textures.verts_features_padded().clone()
    #         self.mesh_texture.textures = TexturesVertex(verts_features=self.sphere_verts_rgb)

    # def _parametrize_pose_batches_2lists(self, r_rotation_batch_all, t_translation_batch_all):
    #     if  r_rotation_batch_all is not None and t_translation_batch_all is not None:
    #         if self.task["pose by texture"] and self.task["fix first pose"]:
    #             self.r_rotation_batch_fixed = r_rotation_batch_all[-1:] # last entry in list corresponds to initial cam position
    #             self.t_translation_batch_fixed = t_translation_batch_all[-1:]
    #             # just applicable if at least >=2 images were already seen
    #             self.r_rotation_batch_param = nn.Parameter(r_rotation_batch_all[0:-1]) # first entry corresponds to latest cam observation!
    #             self.t_translation_batch_param = nn.Parameter(t_translation_batch_all[0:-1])
    #
    #         if self.task["pose by texture"] and self.task["fix first pose"] == False:
    #             self.r_rotation_batch_fixed = None
    #             self.t_translation_batch_fixed = None
    #             # just applicable if at least >=2 images were already seen
    #             self.r_rotation_batch_param = nn.Parameter(r_rotation_batch_all)
    #             self.t_translation_batch_param = nn.Parameter(t_translation_batch_all)
    #
    #         elif self.texture_learning_only:
    #             self.r_rotation_batch_fixed = r_rotation_batch_all
    #             self.t_translation_batch_fixed = t_translation_batch_all
    #             self.r_rotation_batch_param = None
    #             self.t_translation_batch_param = None
    #
    #     else:
    #         self.r_rotation_batch_fixed = r_rotation_batch_all
    #         self.t_translation_batch_fixed = t_translation_batch_all

    def _get_poses_target_images(self, world_point_cloud_map=None):
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
        if self.task["deformation use node point landmarks on world map"]:
            matches_with_world_map_list = []
            points_2D_map_list = []
            split_indices_crossing_list = []
            split_indices_border_list = []


        for i, graph_images in enumerate(self.seen_images[: self.last_images_applied]):
            graph_image = graph_images[2]
            r_rotation_batch[i, :] = graph_image.get_rotation_vector
            t_translation_batch[i, :] = graph_image.get_translation_vector

            _, image_rgb, image_grey = self.UpComingDataGenerator.load_and_create_graph_image_object(
                desired_batch_nr=graph_images[0],
                item_in_batch=graph_images[1],)

            if self.texture_type == TextureTypes.VERTEX_TEXTURE_RGB or self.texture_type == TextureTypes.VERTEX_TEXTURE:
                target_images_list_rgb.append(image_rgb)
            if self.texture_type == TextureTypes.VERTEX_TEXTURE_GREY:
                target_images_list_grey.append(image_grey)
            target_image_mask_list.append(graph_image.generate_mask_for_image(return_as_torch=True))  #ToDo generate mask for target image and rendered world_map seperatly! With node matches!!!

            matching_mask_list.append(graph_image.generate_matching_mask_for_image(return_as_torch=True))
            if PLOT_MASKED_IMAGE_FOR_TESTURE_USAGE:
                if self.texture_type == TextureTypes.VERTEX_TEXTURE_RGB or self.texture_type == TextureTypes.VERTEX_TEXTURE:
                    plot_image_with_mask(image_rgb, graph_image.generate_matching_mask_for_image(return_as_torch=True))
                if self.texture_type == TextureTypes.VERTEX_TEXTURE_GREY:
                    plot_image_with_mask(image_grey, graph_image.generate_matching_mask_for_image(return_as_torch=True), is_grey=True)

            if self.task["deformation use node point landmarks on world map"] or self.task["deformation use world map node points for masking chamfer pcs"]:
                matches_with_world_map_list.append(graph_image.get_matches_to_world_map)
                points_2D_map_list.append(torch.from_numpy(graph_image.get_all_image_nodes).to(device=self.device))
                split_indices_crossing_list.append(graph_image.get_crossing_node_indices_by_removed_border)
                split_indices_border_list.append(graph_image.get_end_node_indices_by_removed_border)


        if self.texture_type == TextureTypes.VERTEX_TEXTURE_RGB or self.texture_type == TextureTypes.VERTEX_TEXTURE:
            self.target_images = torch.tensor(target_images_list_rgb, device=self.device)/255
        if self.texture_type == TextureTypes.VERTEX_TEXTURE_GREY:
            target_image_grey = (torch.tensor(target_images_list_grey, device=self.device)/255)
            self.target_images = torch.stack((target_image_grey, target_image_grey, target_image_grey), dim=3) #target_image_grey in rgb channel!

        self.image_mask = torch.vstack(target_image_mask_list)

        self.matching_image_mask_list = torch.vstack(matching_mask_list)

        if self.task["deformation use node point landmarks on world map"] or self.task["deformation use world map node points for masking chamfer pcs"]:
            self.node_point_container = NodePointContainer(world_point_cloud_map=world_point_cloud_map,
                                                           matches_with_world_map_list=matches_with_world_map_list, points_2D_map_list=points_2D_map_list, split_indices_crossing_list=split_indices_crossing_list, split_indices_border_list=split_indices_border_list)

        return r_rotation_batch, t_translation_batch


    def forward(self, plot_bool:bool=True, iteration:int=0, ith_call:int =0):
        loss = {k: torch.tensor(0.0, device=self.device) for k in self.losses}
        # Deform the mesh with deform_verts_param as parameter
        new_src_mesh = self.geometry_mesh.offset_verts(self.deform_verts_param)
        self.latest_new_src_mesh = new_src_mesh
        # ToDo filtered image (as texture) could also be used! (needs to be tested in a separate loss!!)
        #loss = self.update_mesh_shape_prior_losses(mesh=new_src_mesh, loss=loss)
        loss = update_mesh_shape_prior_losses(mesh=new_src_mesh, loss=loss)

        fine_mesh_texture = subdivideMesh(mesh=new_src_mesh, iter=self.task["subdivide level for textured mesh"])
        fine_mesh_texture.textures = self.texture

        size_all_views = 0
        #ToDo fake the target image for testing somewhere!

        if self.task["deformation use node point landmarks on world map"]:
            loss_image_projection_crossing_nodes, loss_image_projection_end_nodes = self.forward_projection_loss_nodey_keys_in_image_plane(
                new_src_mesh=new_src_mesh, plot_bool=plot_bool, iteration=iteration)
            loss["image reprojection - crossing nodes"] = loss_image_projection_crossing_nodes / 10
            loss["image reprojection - end nodes"] = loss_image_projection_end_nodes / 10


        for j in range(len(self.target_camera_perspective_list)):
            idx_start = j #ToDo atm len_current_batch always need to be exactly 1! so seperating idx_start from j is not beneficial atm (but keeps it general for future adaptions)
            images_rendered = self.renderer_texture_observe(fine_mesh_texture, cameras=self.target_camera_perspective_list[j], lights=self.lights_ambient) #ToDo render a batch >1 at once!
            # Squared L2 distance between the predicted RGB image and the target
            # image from our dataset
            len_current_batch = images_rendered.shape[0]
            images_predicted = images_rendered[..., :3]
            # visualize_image(img=images_predicted)

            if self.task["deformation use chamfer loss"]:
                # def get_point_cloud(img_torch, mask_torch, threshold=0.4):
                #     mask = torch.bitwise_and(torch.gt(img_torch[0, :, :, 0], threshold), mask_torch[0])
                #     idx_matrix = self.idx_matrix[mask]
                #     #idx_matrix = ((img_torch[0, :, :, 0] > 0.2) & (mask_torch[0] == True)).nonzero().squeeze()
                #     return idx_matrix

                if self.task["deformation use chamfer loss"] and self.task[
                    "deformation use chamfer loss in 3D point cloud"]:
                    if iteration==0: #has to be calculated for every image and pose configuration just once!
                        with torch.no_grad():
                            # self.node_point_container.get_matches_with_world_map_list
                            # self.node_point_container.get_updated_node_points_based_on_mesh(geometry_mesh=new_src_mesh)

                            x_point_cloud_predicted = get_point_cloud(img_torch=images_predicted,
                                                                      mask_torch=self.matching_image_mask_list[
                                                                                 idx_start: idx_start + len_current_batch],
                                                                      threshold=self.task["deformation threshold for point cloud extraction"], idx_matrix=self.idx_matrix) #ToDp implement proper masking!! like in 2D!
                            img_predicted_pix_pos_of_patterns = x_point_cloud_predicted.type(dtype=torch.long)
                            img_predicted_pix_pos_of_patterns[:, 1] = img_predicted_pix_pos_of_patterns[:, 1] - 1
                            point_cloud_3D_predicted, fragments_map  =self.inverse_renderer(meshes_world=new_src_mesh, #ToDo check if using the fine_mesh would improve anything here
                                                                     node_pos= img_predicted_pix_pos_of_patterns, cameras= self.target_camera_perspective_list[j],
                                                                     req_gradient=False, req_fragments=True)

                            self.point_cloud_map_texture_feature_mesh = PointCloudMap(point_cloud_3D_predicted,
                                                                       types=None,
                                                                       remove_node_type=None,
                                                                       fragments_map=fragments_map,
                                                                       new_orb_features=None,
                                                                       nx_graph=None, device=self.device
                                                                       )
                            # updated_point_cloud = fragments_map.update_point_cloud(updated_mesh=new_src_mesh)
                            #

                            y_img_target_pix_pos_of_patterns = get_point_cloud(
                                img_torch=self.target_images[idx_start: idx_start + len_current_batch],
                                mask_torch=self.matching_image_mask_list[idx_start: idx_start + len_current_batch],
                                threshold=0.35, idx_matrix=self.idx_matrix)
                            self.y_img_target_pix_pos_of_patterns = y_img_target_pix_pos_of_patterns.type(dtype=torch.long)

                    # after every deformation /iteration
                    predicted_texture_points_based_on_new_deformed_mesh = self.point_cloud_map_texture_feature_mesh.update_point_cloud_based_on_mesh(
                        mesh=new_src_mesh)

                    y_img_target_pix_pos_of_patterns = self.y_img_target_pix_pos_of_patterns
                    point_cloud_3D_target_image_patterns, _ = self.inverse_renderer(meshes_world=new_src_mesh,
                                                                                         # ToDo check if using the fine_mesh would improve anything here
                                                                                         node_pos=y_img_target_pix_pos_of_patterns,
                                                                                         cameras=
                                                                                         self.target_camera_perspective_list[
                                                                                             j],
                                                                                         req_gradient=True,
                                                                                         req_fragments=False)

                    error, _ = chamfer_distance(x=predicted_texture_points_based_on_new_deformed_mesh.points_packed()[None], y=point_cloud_3D_target_image_patterns.points_packed()[None])
                    loss_error_chamfer_distance = error
                    loss["texture chamfer loss"] += loss_error_chamfer_distance * len_current_batch

                if self.task["deformation use chamfer loss"] and self.task[
                    "deformation use chamfer loss in 2D in image plane"]:

                    if iteration==0: # has to be calculated for every image and pose configuration just once!
                        with torch.no_grad():
                            if self.task["deformation use world map node points for masking chamfer pcs"]:
                                match_idcs_with_world_map = self.node_point_container.get_matches_with_world_map_list[j][:, 0]
                                key_node_points_world_map_3d = self.node_point_container.get_updated_node_points_based_on_mesh(
                                    geometry_mesh=new_src_mesh)[match_idcs_with_world_map]
                                re_projected_key_node_points_world_map_2d = project_3d_points_to_image_plane_without_distortion(
                                    self.projection_matrices[j], key_node_points_world_map_3d,
                                    image_size=self.image_size_torch_tensor)
                                mask_key_matches_predicted_image = generate_mask(node_pos=re_projected_key_node_points_world_map_2d.cpu().numpy().astype(int),
                                    mask_size=(self.target_images.shape[1], self.target_images.shape[1]),
                                    return_as_torch=True, device=self.device)
                            else:
                                mask_key_matches_predicted_image = torch.ones(1, self.target_images.shape[1], self.target_images.shape[1]).to(dtype=torch.bool, device=self.device)

                            if self.task["deformation skeletonize predicted image before point cloud extraction"]:
                                img_np = images_predicted[0, :, :, 0].clone().detach().cpu().numpy()
                                image_grey_distorted = img_np*255
                                skeletonized_image_grey = skeletonise_and_clean(thr_image=image_grey_distorted, plot=True, save=False, directory='')
                                img_torch = torch.stack((torch.tensor(skeletonized_image_grey), torch.tensor(skeletonized_image_grey),torch.tensor(skeletonized_image_grey)), dim=2)[None].to(device=self.device)
                                #
                                # x_point_cloud_predicted = get_point_cloud(img_torch=img_torch,
                                #                                           mask_torch=mask_key_matches_predicted_image,
                                #                                           threshold=0.42, idx_matrix=self.idx_matrix)
                            else:
                                img_torch = images_predicted
                            x_point_cloud_predicted = get_point_cloud(img_torch=img_torch,
                                                                      mask_torch=mask_key_matches_predicted_image,
                                                                      threshold=self.task["deformation threshold for point cloud extraction"], idx_matrix=self.idx_matrix)

                            ##ToDo

                            self.y_img_target_pix_pos_of_patterns = get_point_cloud(img_torch=self.target_images[idx_start: idx_start + len_current_batch],
                                                                               mask_torch=self.matching_image_mask_list[idx_start: idx_start + len_current_batch], threshold=0.35, idx_matrix=self.idx_matrix)

                            #y_img_target_pix_pos_of_patterns = y_img_target_pix_pos_of_patterns - 1
                            #y_img_target_pix_pos_of_patterns = get_point_cloud(img_torch=self.target_images[-1:], mask_torch=self.matching_image_mask_list[-1:], threshold=0.25)
                            #x_point_cloud_predicted = x_point_cloud_predicted-1
                            x_point_cloud_predicted[:, 1] = x_point_cloud_predicted[:, 1]-1 #to compensate for constant dismatch for an 128x128 image
                            img_predicted_pix_pos_of_patterns = x_point_cloud_predicted.type(dtype=torch.long)
                            # img_predicted_pix_pos_of_patterns_in = torch.stack((img_predicted_pix_pos_of_patterns[:, 1], img_predicted_pix_pos_of_patterns[:,0])).t()
                            # img_predicted_pix_pos_of_patterns_in = img_predicted_pix_pos_of_patterns
                            texture_points_3d, fragments_map = self.inverse_renderer(meshes_world=new_src_mesh, #ToDo check if using the fine_mesh would improve anything here
                                                                         node_pos= img_predicted_pix_pos_of_patterns, cameras= self.target_camera_perspective_list[j],
                                                                         req_gradient=True, req_fragments=True)
                            self.point_cloud_map_texture_feature_mesh = PointCloudMap(texture_points_3d,
                                                                       types=None,
                                                                       remove_node_type=None,
                                                                       fragments_map=fragments_map,
                                                                       new_orb_features=None,
                                                                       nx_graph=None, device=self.device)
                            # to define corresponding point cloud on mesh!
                    # after every deformation /iteration
                    predicted_texture_points_based_on_new_deformed_mesh = self.point_cloud_map_texture_feature_mesh.update_point_cloud_based_on_mesh(
                        mesh=new_src_mesh)
                    y_img_target_pix_pos_of_patterns = self.y_img_target_pix_pos_of_patterns
                    projected_image_keypoints_2d = camera_projection(predicted_texture_points_based_on_new_deformed_mesh.points_packed(), camera=self.target_camera_perspective_list[j],
                                                                     image_size=self.image_size_torch_tensor, device=self.device, r_rotation=None, t_translation=None)
                    #projected_image_keypoints_2d[:, 1] = projected_image_keypoints_2d[:, 1]-1 #to compensate for constant dismatch for an 128x128 image
                    error, _ = chamfer_distance(x=projected_image_keypoints_2d[None], y=y_img_target_pix_pos_of_patterns[None])
                    loss_error_chamfer_distance = error
                    loss["texture chamfer loss"] += loss_error_chamfer_distance * len_current_batch
                # error, _ = chamfer_distance(x=y_img_target_pix_pos_of_patterns[None],
                #                             y=projected_image_keypoints_2d[None])

                # visualize_features_on_image(image=images_predicted,
                #                             pos1=y_img_target_pix_pos_of_patterns,
                #                             pos2=projected_image_keypoints_2d, node_thickness=30,
                #                             title='Features on Texture Model with Extracted and Target Features')
                # visualize_features_on_image(image=images_predicted,
                #                             pos1=img_predicted_pix_pos_of_patterns,
                #                             node_thickness=30,
                #                             title='Features on Texture Model with Extracted and Target Features')
                #
                # visualize_features_on_image(image=images_predicted,
                #                             node_thickness=30,
                #                             title='Features on Texture Model with Extracted and Target Features')
                #
                # visualize_features_on_image(image=self.target_images[idx_start: idx_start + len_current_batch],
                #                             pos1=y_img_target_pix_pos_of_patterns, node_thickness=30,
                #                             title='Matching Features on Target Image')
                #
                # visualize_features_on_image(image=self.target_images[-1:],
                #                             pos1=y_img_target_pix_pos_of_patterns, node_thickness=30,
                #                             title='Matching Features on Target Image')

                if (self.task["deformation use chamfer loss - make plots"] and iteration % self.task["use chamfer loss - make plots - all variations in period of"] == 0) or self.task['deformation - visualization of chamfer loss every iteration for gif building']:

                    if self.task["deformation use chamfer loss in 3D point cloud"]:
                        projected_image_keypoints_2d = camera_projection(predicted_texture_points_based_on_new_deformed_mesh.points_packed(), camera=self.target_camera_perspective_list[j],
                                                                         image_size=self.image_size_torch_tensor, device=self.device, r_rotation=None, t_translation=None)
                    # Remember, the projected_image_keypoints_2d don't align up perfectly with the image,
                    fig_chamfer_pcs_pose_recon, ax_chamfer_pcs_pose_recon = visualize_features_on_image(image=images_predicted, pos1=projected_image_keypoints_2d, pos2=y_img_target_pix_pos_of_patterns,
                                                                                                        node_thickness=3* SCALE_FIGURE_SETTINGs, title='Features on Texture Model with Extracted and Target Features',
                                                                                                        size_width=3.25, size_height=3.25)
                    if self.task["deformation use node point landmarks on world map"]:
                        target_image_nodes = self.target_node_keypoints_2d.cpu().detach().numpy()
                        predicted_image_nodes = self.re_projected_image_keypoints_2d.cpu().detach().numpy()
                        for i in range(target_image_nodes.shape[0]):
                            x = (target_image_nodes[i, 0], predicted_image_nodes[i, 0])
                            y = (target_image_nodes[i, 1], predicted_image_nodes[i, 1])
                            # Plot the connecting lines
                            ax_chamfer_pcs_pose_recon.plot(x, y, color=red_3, alpha=1, linewidth=3 * SCALE_FIGURE_SETTINGs)

                    fig_chamfer_pcs_pose_recon, ax_chamfer_pcs_pose_recon = keep_grid_and_remove_ticks_and_labels(fig=fig_chamfer_pcs_pose_recon, ax=ax_chamfer_pcs_pose_recon)
                    fig_chamfer_pcs_pose_recon.show()

                    if self.task["deformation use chamfer loss - make plots"] and iteration % self.task[
                        "use chamfer loss - make plots - all variations in period of"] == 0:
                        save_figure(fig=fig_chamfer_pcs_pose_recon, ax = ax_chamfer_pcs_pose_recon, show_fig=True,
                                    name_of_figure='Deformation chamfer_loss_point_cloud_alignment_on_rendered_image_call' + str(ith_call) + '_itr_' + str(iteration))

                    if self.task['deformation - visualization of chamfer loss every iteration for gif building']:
                        self.Gif_deformation_chamfer_loss_texture.add_figure(fig=fig_chamfer_pcs_pose_recon)



                    self.visualize_mesh(title='Mesh with texture', batch_size=1, if_save=False, make_gif=self.task['deformation - visualization of chamfer loss every iteration for gif building'])


                    #ax_chamfer_pcs_pose_recon.axis('off')

                    if self.task["deformation use chamfer loss - make plots - all variations"]:

                        fig, ax =visualize_features_on_image(image=self.target_images[idx_start: idx_start + len_current_batch],
                                                    pos1=y_img_target_pix_pos_of_patterns, pos2=projected_image_keypoints_2d, node_thickness=20, title='Matching Features on Target Image with Predicted Backprojected Features')
                        fig.show()
                        fig, ax =visualize_features_on_image(image=images_predicted,
                                                    pos1=img_predicted_pix_pos_of_patterns, node_thickness=25, title='Features on Texture Model - Predicted Image')
                        fig.show()

                        fig, ax =visualize_features_on_image(image=self.target_images[idx_start: idx_start + len_current_batch],
                                                    pos1=y_img_target_pix_pos_of_patterns, node_thickness=20, title='Matching Features on Target Image')
                        fig.show()

            if self.task["deformation use texture comparison"]:
                images_rendered_pattern_high_diff = self.renderer_texture_observe(fine_mesh_texture, cameras=self.target_camera_perspective_list[j],
                                              lights=self.lights_ambient)
                images_predicted_pattern_high_diff = images_rendered_pattern_high_diff[..., :3]
                #visualize_image(img=images_predicted_pattern_high_diff)
                error = images_predicted_pattern_high_diff - self.target_images[idx_start: idx_start + len_current_batch]
                error = error[self.matching_image_mask_list[idx_start: idx_start + len_current_batch]]
                loss_error_texture = (error ** 2).mean()
                loss["pure texture"] += loss_error_texture * len_current_batch

            size_all_views += len_current_batch

        loss["texture chamfer loss"] = (loss["texture chamfer loss"]/size_all_views) /10
        loss["pure texture"] = (loss["pure texture"] / size_all_views)/ 10000

        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=self.device)
        for k, l in loss.items():
            sum_loss += l * self.losses[k]["weight"]
            self.losses[k]["values"].append(float(l.detach().cpu())*self.losses[k]["weight"])

        return sum_loss, new_src_mesh

    def forward_projection_loss_nodey_keys_in_image_plane(self, new_src_mesh, plot_bool, iteration: int = 0):
        assert self.projection_matrices is not None
        world_points_based_on_new_mesh = self.node_point_container.get_updated_node_points_based_on_mesh(geometry_mesh=new_src_mesh)

        if self.node_point_container.does_node_container_handle_splits():
            loss_image_projection_vector_crossing_nodes = torch.zeros(self.node_point_container.get_points_2D_map_list.__len__(), dtype=torch.float32,
                                                                      device=self.device)
            loss_image_projection_vector_end_nodes = torch.zeros(self.node_point_container.get_points_2D_map_list.__len__(), dtype=torch.float32,
                                                                 device=self.device)
        # updated_point_cloud = self.world_map.world_point_cloud_map.update_point_cloud_based_on_mesh(mesh=mesh)
        # world_points_based_on_new_mesh = updated_point_cloud.points_packed()

        for i in range(self.node_point_container.get_points_2D_map_list.__len__()):
            key_node_points_world_map_3d = world_points_based_on_new_mesh[self.node_point_container.get_matches_with_world_map_list[i][:, 0]]
            re_projected_image_keypoints_2d = project_3d_points_to_image_plane_without_distortion(self.projection_matrices[i], key_node_points_world_map_3d, image_size=self.image_size_torch_tensor)

            target_node_keypoints_2d = self.node_point_container.get_points_2D_map_list[i][self.node_point_container.get_matches_with_world_map_list[i][:, 1]]

            # target_image_rgb = self.target_images_and_pos[i][1]
            # # not needed here since we are using the predicted node pos at that point!
            # original_image_keypoints_2d = self.image_keypoints_2d_list[i][0][self.matches_with_world_map[i][:, 1]]
            loss_image_projection, error_residual = calc_reprojection_error_matrix(key_node_points_world_map_3d,
                                                                                   target_node_keypoints_2d, proj_matricies=self.projection_matrices[i],
                                                                                   image_size=self.image_size_torch_tensor)

            loss_image_projection_vector_crossing_nodes[i] = error_residual[self.node_point_container.get_split_indices_crossing_list[i]].mean()
            loss_image_projection_vector_end_nodes[i] = error_residual[self.node_point_container.get_split_indices_border_list[i]].mean()

            # for plotting!
            self.target_node_keypoints_2d = target_node_keypoints_2d
            self.re_projected_image_keypoints_2d = re_projected_image_keypoints_2d

            # image_size = self.image_size[0].cpu().numpy()
            # if i == self.plot_image_nr and plot_bool:
            #     predicted_nodes_np = re_projected_image_keypoints_2d.clone().cpu().detach().numpy()
            #     target_nodes_np = target_node_keypoints_2d.clone().cpu().detach().numpy()
            #     group_of_nodes = {
            #         'labels': ['target crossing node', 'target end node', 'updated crossing node', 'updated end node'],
            #         'color': [NodeTypes.END.colour, NodeTypes.CROSSING.colour,
            #                   NodeTypes.END.colour, NodeTypes.CROSSING.colour],
            #         'data': [[target_nodes_np[indices_crossing_nodes, 0],
            #                   image_size - target_nodes_np[indices_crossing_nodes, 1]],
            #                  [target_nodes_np[indices_end_nodes, 0],
            #                   image_size - target_nodes_np[indices_end_nodes, 1]],
            #                  [predicted_nodes_np[indices_crossing_nodes, 0],
            #                   image_size - predicted_nodes_np[indices_crossing_nodes, 1]],
            #                  [predicted_nodes_np[indices_end_nodes, 0],
            #                   image_size - predicted_nodes_np[indices_end_nodes, 1]]],
            #         'alpha': [1, 1, 0.6, 0.6],
            #         'marker': ["*", "*", ".", "."]
            #         }
            #     fig_deformation_error = plot_deformation_error(group_of_nodes=group_of_nodes,
            #                                                    target_nodes_np=target_nodes_np,
            #                                                    predicted_nodes_np=predicted_nodes_np,
            #                                                    image_size=image_size)
            #
            #     save_figure(fig=fig_deformation_error,
            #                 name_of_figure='deformation_error_' + str(self.ith_call) + '_ith_call' + str(
            #                     iteration) + '_ith_it')
            #
            #     fig_deformation_error_on_graph = plot_deformation_on_graph(group_of_nodes=group_of_nodes,
            #                                                                target_nodes_np=target_nodes_np,
            #                                                                predicted_nodes_np=predicted_nodes_np,
            #                                                                image_size=image_size,
            #                                                                adjacency=self.world_map.world_point_cloud_map.nx_graph.adj_matrix_square,
            #                                                                image_rgb=None,
            #                                                                title='Ratio between target and updated node position ' + "\n" + ' iteration call: ' + str(
            #                                                                    iteration))
            #     save_figure(fig=fig_deformation_error_on_graph,
            #                 name_of_figure='deformation_error_on_graph_' + str(self.ith_call) + '_ith_call' + str(
            #                     iteration) + '_ith_it')
            #
            #     fig_deformation_error_on_graph_image = plot_deformation_on_graph(group_of_nodes=group_of_nodes,
            #                                                                      target_nodes_np=target_nodes_np,
            #                                                                      predicted_nodes_np=predicted_nodes_np,
            #                                                                      image_size=image_size,
            #                                                                      adjacency=self.world_map.world_point_cloud_map.nx_graph.adj_matrix_square,
            #                                                                      image_rgb=target_image_rgb,
            #                                                                      title='Ratio between target and updated node position ' + "\n" + ' iteration call: ' + str(
            #                                                                          iteration))
            #     save_figure(fig=fig_deformation_error_on_graph_image,
            #                 name_of_figure='deformation_error_on_graph_image_' + str(self.ith_call) + '_ith_call' + str(
            #                     iteration) + '_ith_it')
            #
            #     self.GifMakerGraphImage.add_figure(fig=fig_deformation_error_on_graph_image)
            #     fig_deformation_error_on_image = plot_deformation_on_graph(group_of_nodes=group_of_nodes,
            #                                                                target_nodes_np=target_nodes_np,
            #                                                                predicted_nodes_np=predicted_nodes_np,
            #                                                                image_size=image_size,
            #                                                                adjacency=self.world_map.world_point_cloud_map.nx_graph.adj_matrix_square,
            #                                                                image_rgb=target_image_rgb, plot_graph=False,
            #                                                                title='Ratio between target and updated node position ' + "\n" + ' iteration call: ' + str(
            #                                                                    iteration))
            #
            #     save_figure(fig=fig_deformation_error_on_image,
            #                 name_of_figure='deformation_error_on_image_' + str(self.ith_call) + '_ith_call' + str(
            #                     iteration) + '_ith_it')
            #
            #     self.GifMakerImage.add_figure(fig=fig_deformation_error_on_image)

        return loss_image_projection_vector_crossing_nodes.mean(), loss_image_projection_vector_end_nodes.mean()

    def slice_batch_to_camera_lists(self, r_rotation_batch, t_translation_batch):
        target_camera_for_optimization = []
        for i in range(0, r_rotation_batch.shape[0], self.task["deformation batch size for rendering"]):
            rotation_batch_sliced = r_rotation_batch[
                                    i: min(i + self.task["deformation batch size for rendering"], r_rotation_batch.shape[0])]
            t_translation_batch_sliced = t_translation_batch[
                                         i: min(i + self.task["deformation batch size for rendering"], r_rotation_batch.shape[0])]
            R_batch = self.cam_rodrigues_object.get_rotation_matrix(rotation_batch_sliced)
            T_batch = self.cam_rodrigues_object.get_translation_matrix(t_translation_batch_sliced)
            target_camera_for_optimization.append(PerspectiveCameras(device=self.device, R=R_batch,
                                                                     T=T_batch, focal_length=self.focal_length))
        return target_camera_for_optimization

    def update_losses(self, losses):
        self.losses = losses

    def update_task(self, task):
        if task is not None:
            self.task.update(task)

    def visualize_texture_prediction(self, title:str= 'texture fitting vs. target image'):
        fig, axs = visualize_texture_prediction(textured_mesh=self.get_textured_mesh(),
                                     renderer=self.renderer_texture_observe,
                                     camera_perspective=self.target_camera_perspective_list[0],  # ToDo vary view that is compared!
                                     image_to_compare_torch=self.target_images[0],
                                     title=title, grey=self.texture_type == TextureTypes.VERTEX_TEXTURE_GREY)
        fig.show()
        save_figure(fig=fig, name_of_figure='mesh_view_perspective_rendering_after_deformation_texture fitting vs. target image')

    def close_existing_gifs(self):
        if self.task['deformation - visualization of chamfer loss every iteration for gif building']:
            self.Gif_deformation_chamfer_loss_texture.close_writer()
            self.Gif_deformation_chamfer_loss_view.close_writer()

    def visualize_mesh(self, title:str= 'Model with texture', batch_size=10, if_save=True, make_gif=False):

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
                           title=title, grey=self.texture_type==TextureTypes.VERTEX_TEXTURE_GREY)

            ax.title.set_visible(False)
            if if_save:
                fig.show()
                save_figure(fig=fig, name_of_figure='mesh_view_perspective_rendering_after_deformation_elev_' + str(elev[i].numpy())+ '_azim_' + str(azim[i].numpy()))
            if make_gif and i == 0: #Todo vary i
                self.Gif_deformation_chamfer_loss_view.add_figure(fig=fig)

    def get_rotation_translation(self):
        rotation = None
        translation = None
        if self.r_rotation_batch_fixed  is not None and self.t_translation_batch_fixed is not None:
            rotation = self.r_rotation_batch_fixed
            translation = self.t_translation_batch_fixed
        if self.r_rotation_batch_param is not None and self.t_translation_batch_param is not None:
            self.r_rotation_batch_param.requires_grad = False
            self.t_translation_batch_param.requires_grad = False
            if rotation is not  None and translation is not None:
                rotation = torch.vstack((self.r_rotation_batch_param, rotation))
                translation = torch.vstack((self.t_translation_batch_param, translation))
            else:
                rotation = self.r_rotation_batch_param
                translation = self.t_translation_batch_param
        return rotation, translation

    @property
    def get_idx_of_last_image_applied(self):
        return self.last_images_applied

    def get_textured_mesh(self):
        #mesh = self.geometry_mesh
        #self.mesh_texture.textures.verts_features_padded().requires_grad = False
        fine_mesh_texture = subdivideMesh(mesh=self.latest_new_src_mesh, iter=self.task["subdivide level for textured mesh"])
        fine_mesh_texture.textures = self.texture
        return fine_mesh_texture

    @property
    def get_loss_dict_history(self):
        return self.losses

    @property
    def get_fig_chamfer_loss_reconstruction(self):
        return self.fig_chamfer_loss_pcs_pose_recon

def _get_projection_matrix(camera: PerspectiveCameras, image_size_torch_batch: torch.Tensor = torch.tensor([128, 128])[None]):
    R_cv, t_cv, K_cv = opencv_from_cameras_projection(cameras=camera, image_size=image_size_torch_batch) # ToDo check if K_cv makes sense (shift of principle points)
    torch_cam_3d_batch = [CameraTorch(R_cv=R_cv[i], t_cv=t_cv[i][None], K=K_cv[i],
                                       image_size=image_size_torch_batch, device=R_cv.device) for i in range(R_cv.shape[0])]
    proj_matrix_batch = [torch_cam_3d_batch[i].projection for i in range(R_cv.shape[0])]
    # This camera is using the cv literature convention --> R,T,K needs to be converted!
    return torch.stack(proj_matrix_batch)

def update_mesh_shape_prior_losses(mesh, loss):
    # looks for smoothness, so for a perfect sphere the loss is zero and the loss goes up later
    # Losses to smooth / regularize the mesh shape
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)
    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)
    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")

    return loss

def visualize_mesh(textured_mesh, renderer, camera_perspective, title='',
                         silhouette=False, grey =False, size_width=3.5, size_height=3.5):
    inds = 3 if silhouette else range(3)
    #inds = 0 if grey else range(3)
    with torch.no_grad():
        predicted_images = renderer(textured_mesh, cameras=camera_perspective)
        img = predicted_images.detach().cpu().numpy()[0, :, :, :3]

    fig, ax = generate_mpl_figure(size_width=size_width, size_height=size_height)
    ax.imshow(img)

    ax.axis('off')
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax

# def visualize_mesh(textured_mesh, renderer, camera_perspective, title='',
#                          silhouette=False, grey =False):
#     inds = 3 if silhouette else range(3)
#     #inds = 0 if grey else range(3)
#     with torch.no_grad():
#         # camera_perspective2 =PerspectiveCameras(device='cuda', R=camera_perspective.R.clone(), T=camera_perspective.T.clone(),
#         #                    focal_length=10)
#         predicted_images = renderer(textured_mesh, cameras=camera_perspective)
#     plt.figure(figsize=(20, 20))
#     plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())
#     plt.title(title)
#     plt.axis("off")
#     plt.show()



def visualize_texture_prediction(textured_mesh, renderer, camera_perspective,
                         image_to_compare_torch, title='',
                         silhouette=False, grey = False, size_width=3.25, size_height=3.25):
    #inds = 3 if silhouette else range(3)

    with torch.no_grad():
        predicted_images = renderer(textured_mesh, cameras=camera_perspective)
        img = predicted_images.detach().cpu().numpy()[0, :, :, :3]

    #fig, ax = generate_mpl_figure(size_width=size_width, size_height=size_height)

    # using the variable axs for multiple Axes

    plt = get_plt_style()
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    #fig, ax = plt.subplots(constrained_layout=True)
    fig = size_fig(fig, size_width, size_height)

    axs[0].imshow(img)
    axs[1].imshow(image_to_compare_torch.cpu().detach().numpy())
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[0].axis('off')

    fig.tight_layout()

    return fig, axs

def visualize_features_on_image(image, pos1: np.ndarray =None, pos2: np.ndarray = None,
                                node_thickness:int=15, title: str='Features on Image', size_width=7.5, size_height=7.5):
    if torch.is_tensor(image):
        img = image.detach().cpu().numpy()[0]
    else:
        img = image  # needs to be a numpy array!

    fig, ax = generate_mpl_figure(size_width=size_width, size_height=size_height)
    ax.imshow(img)
    if torch.is_tensor(pos1):
        pos1 = pos1.detach().cpu().numpy()
    if pos1 is not None:
        ax.scatter(x=pos1[:, 1], y=pos1[:, 0], s=node_thickness, c=blue_2, alpha=0.6)
    if torch.is_tensor(pos2):
        pos2 = pos2.detach().cpu().numpy()
    if pos2 is not None:
        ax.scatter(x=pos2[:, 1], y=pos2[:, 0], s=node_thickness, c=red_1, alpha=0.6)

    #ax.axis('on')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
    #fig.show()

    return fig, ax

