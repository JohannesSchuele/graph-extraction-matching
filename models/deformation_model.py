import torch
import torch.nn as nn
import networkx as nx
import cv2
from models.pose_rodrigues_rot_fromula import Camera3DPoseRodriguesRotFormula
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from models.geometry_losses import *

from models.point_cloud_features import PointCloudMap
from configs.plot.config_plots import *
from configs.plot.colours import *
from mpl_toolkits.mplot3d import Axes3D
from models.pc_alignment_pose_opt import scale_point_cloud
from configs.config import *
from utils.world_map_visualization import plot_mesh
from utils.mesh_operations import subdivideMesh

from models.camera_pose import camera_projection
from utils.deformation_helpers import get_index_matrix, get_point_cloud
from tools_generate.image_processing import skeletonise_and_clean


class DeformationModel(nn.Module):
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
    def __init__(self, RenderersCollection, geometry_mesh, losses, focal_length,
                 task, camera_model_view, device="cpu", image_size=None,
                 plot_image_nr: int = 0, ith_call: int = 0, depth_map_dim=(128, 128),
                 depth_map_data=None, texture=None):
        super().__init__()

        self.task = task
        self.RenderersCollection = RenderersCollection
        self.inverse_renderer = RenderersCollection.inverse_renderer
        self.src_mesh = geometry_mesh
        self.device = device
        self.focal_length = focal_length
        self.image_size = image_size
        self.plot_image_nr = plot_image_nr
        self.depth_map_dim = depth_map_dim
        self._all_image_node_pos = None
        self.R_pc_alignment = None
        self.T_pc_alignment = None
        self.scale_xy_plane_alignment = None
        self.camera_model_view = camera_model_view

        self.depth_map_data = depth_map_data
        self.renderer_texture_observe = self.RenderersCollection.renderer_texture
        self.lights_ambient = self.RenderersCollection.lights_ambient
        self.texture = texture

        # image features [manually selected points, skeletons]
        self.feature_pc_previous_map_object = [None, None]
        self.feature_pc_targets = [None, None]

        if losses:
            self.losses = losses
        else:
            self.losses = {"image reprojection - crossing nodes": {"weight": 1.8, "values": []},
                 "image reprojection - end nodes": {"weight": 0.9, "values": []},
                 "edge": {"weight": 0.4, "values": []},
                 "normal": {"weight": 0.4, "values": []},
                 "laplacian": {"weight": 0.4, "values": []},
                 "outlier": {"weight": 0.1, "values": []},
                 "chamfer distance to points": {"weight": 0.8, "values": []},
                 "euclidean distance to points": {"weight": 0.8, "values": []},
                 "image rgb data": {"weight": 0.1, "values": []},
                 "texture features": {"weight": 0.1, "values": []},
                 }
        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=self.device)
        self.proj_matrix_batch = None
        self.__define_parameters()

        self.ith_call = ith_call
        self.reference_mesh4plot = None

        if self.task['deformation - make gif of mesh model driven by the depth map']:
            self.GifMakerImageDepthMapMesh = GifMaker(name_of_gif='deformation_error_on_detph_map_mesh'+str(ith_call)+'_ith_call')
            self.GifMakerImageDepthMapPointCloud = GifMaker(name_of_gif='deformation_error_on_detph_map_point_cloud'+str(ith_call)+'_ith_call')
            self.GifMakerImageDepthMapTexture = GifMaker(name_of_gif='deformation_texture_feature_error'+str(ith_call)+'_ith_call')
            # self.GifMakerGraphImage = GifMaker(name_of_gif='deformation_error_on_graph_image_' + str(ith_call)
            #                                               +'_ith_call')
        #self.compare_to_point_cloud = self.losses['chamfer distance to points']['weight'] > 0.0
        if self.task['deformation - depth map optimization']:

            #self.inital_cam_depth_map_perspective_model = depth_map_data.synthetic_camera
            #Todo: generalize R and T also as an argument?!!

            # Initial rot & trnsl. for point cloud based on rotation of camera (in data container) and translation of 1
            self.R_measurement_pc, self.T_measurement_pc = self.depth_map_data.initial_point_cloud_alignment
            # point cloud rendered of source mesh from synth. cam perspective
            self.point_cloud_map = self.define_point_cloud(mesh=self.src_mesh, R=self.camera_model_view.R, T=self.camera_model_view.T,
                                                           image_dim=self.depth_map_dim)  # pc of mesh shape in view

            losses_pc_alignment = {"euclidean distance": {"weight": 1.0, "values": []},
                           "scale xy-plane ": {"weight": 1.7, "values": []}}


            if self.task['deformation - depth map optimization - align point cloud on mesh']:
                R_aligmnet_optimization, T_alignment_optimization, scale_xy_plane = self.align_point_cloud_on_mesh(
                    mesh=self.src_mesh,
                    R_measurement_pc_init=self.R_measurement_pc,
                    T_measurement_pc_init=self.T_measurement_pc,
                    node_pos_key=depth_map_data.measurement_node_pos_key,
                    point_cloud_keys_2b_aligned=depth_map_data.measurement_key_point_cloud,
                    losses_pc_alignment=losses_pc_alignment,
                    overall_point_clouds=[self.point_cloud_map.points_packed()[None],
                                          depth_map_data.measurement_point_cloud])

                point_cloud_target_init_pose = depth_map_data.measurement_point_cloud
                scaled_point_cloud_init_pose = scale_point_cloud(torch_point_cloud=point_cloud_target_init_pose,
                                                                 scale_xy_plane=scale_xy_plane)
                self.R_pc_alignment = R_aligmnet_optimization
                self.T_pc_alignment = T_alignment_optimization
                self.scale_xy_plane_alignment = scale_xy_plane
                self.R_measurement_pc = R_aligmnet_optimization
                self.T_measurement_pc = T_alignment_optimization
            else:
                scaled_point_cloud_init_pose = depth_map_data.scaled_measurement_point_cloud

            self.point_cloud_target = torch.add(torch.matmul(scaled_point_cloud_init_pose,
                                                self.R_measurement_pc), self.T_measurement_pc)

        if self.task['deformation - texture optimization']:
            self.idx_matrix = get_index_matrix(device=self.device, image_size_int=image_size)

    def __define_parameters(self):
        # deform the mesh by offsetting its vertices
        # The shape of the deform parameters is equal to the total number of vertices of the mesh
        self.verts_shape = self.src_mesh.verts_packed().shape
        deform_verts = torch.full(self.verts_shape, 0.0, device=self.device, requires_grad=True)
        self.deform_verts_param = nn.Parameter(deform_verts)


    def forward(self, plot_bool=False, iteration:int=0, ith_call:int =0, plot_different_perspectives=False):
        loss = {k: torch.tensor(0.0, device=self.device) for k in self.losses}
        # Deform the mesh with deform_verts_param as parameter
        new_src_mesh = self.src_mesh.offset_verts(self.deform_verts_param)
        # ToDo filtered image (as texture) could also be used! (needs to be tested in a separate loss!!)
        #loss = self.update_mesh_shape_prior_losses(mesh=new_src_mesh, loss=loss)
        #def f(iteration, n_iterations=150, min_value=temp_min_val):
        #    factor = 1-((1-min_value)*iteration)/n_iterations
        #    print(factor)
        #    return factor
        loss = update_mesh_shape_prior_losses(mesh=new_src_mesh, loss=loss, #factor=f(iteration),
            edge_consistency_loss=self.task['deformation - depth map optimization - edge consistency loss else edge loss'])
        # loss['outlier'] = mesh_delta_edge_length_loss(new_src_mesh)

        # ----------------------------------------------------------------------
        # Depth map optimization
        # ----------------------------------------------------------------------

        if self.task['deformation - depth map optimization']:
            # Calculate point cloud
            if self.task['deformation - depth map optimization - rerender target point cloud over iterations']:
                self.point_cloud_map = self.define_point_cloud(mesh=new_src_mesh, R=self.camera_model_view.R, T=self.camera_model_view.T,
                                                               image_dim=self.depth_map_dim) # ToDo NaN value check
            else:
                self.point_cloud_map.update_point_cloud_based_on_mesh(
                    mesh=new_src_mesh)
            mask = torch.logical_not(torch.isnan(self.point_cloud_map.points_packed()[None]))

            # mesh reference for plot 2 include color based on length changes
            if iteration == 0:
                self.reference_mesh4plot = new_src_mesh

            # Plots of deformations
            if self.task['deformation - depth map optimization - plot deformation'] and iteration % self.task['deformation - depth map optimization - plot period'] == 0:
                # Plot of mesh either with point cloud or colored based on change to mesh edge lengths
                if not self.task['deformation - depth map optimization - plot mesh stress']:
                    # plot entire mesh with point cloud
                    fig_point_cloud_on_mesh_model, ax = generate_mpl_3D_figure(size_width=3.5, size_height=3.5, do_tight_fit=False)
                    ax.set_xlim([-1, 1])
                    ax.set_ylim([-1, 1])
                    ax.set_zlim([-1, 1])
                    fig_point_cloud_on_mesh_model, ax = plot_single_point_cloud(torch_point_cloud=self.point_cloud_target[:, mask[0, :, 0], :],
                                                      title='compare current shape to target',
                                                      ax=ax, fig=fig_point_cloud_on_mesh_model, color=red_3, alpha=0.35, thickness=0.02)
                    fig_point_cloud_on_mesh_model, ax = plot_mesh(mesh_model=new_src_mesh, fig=fig_point_cloud_on_mesh_model,
                                                                  ax=ax, color_nodes=green_6, alpha_nodes =0.75,
                                                                  thickness_nodes=0.45*SCALE_FIGURE_SETTINGs,
                                                                  color_edges=green_6, alpha_edges=0.6)
                    ax.set_axis_off()
                    ax.view_init(elev=25, azim=35)
                    fig_point_cloud_on_mesh_model.show()
                    ax.title.set_visible(False)
                    save_figure(fig=fig_point_cloud_on_mesh_model, name_of_figure='deformation 3D mesh - depth map iteration_'+str(iteration)+'_th_call_'+str(ith_call), do_tight_fit=False)
                    if self.task['deformation - make gif of mesh model driven by the depth map']:
                        self.GifMakerImageDepthMapMesh.add_figure(fig_point_cloud_on_mesh_model)
                else:
                    # plot changes of mesh edge lengths (using colormap)
                    fig_mesh_edge_change, ax = generate_mpl_3D_figure(size_width=3.5, size_height=3.5, do_tight_fit=False)
                    ax.set_xlim([-1, 1])
                    ax.set_ylim([-1, 1])
                    ax.set_zlim([-1, 1])
                    fig_mesh_edge_change, ax = plot_mesh(
                        mesh_model=new_src_mesh, fig=fig_mesh_edge_change, ax=ax,
                        linewidth=0.5*SCALE_FIGURE_SETTINGs, alpha_edges=0.5,
                        reference_mesh4color=self.reference_mesh4plot, focus_on='retraction',
                        cut_off_plane=(torch.Tensor([0, 0, 0.25]), torch.Tensor([0, 0, 1])))
                    ax.set_axis_off()
                    ax.view_init(elev=75, azim=0)
                    fig_mesh_edge_change.show()
                    ax.title.set_visible(False)
                    save_figure(fig=fig_mesh_edge_change,
                                name_of_figure='deformation mesh edge change - depth map iteration_'+str(iteration) +
                                               '_th_call_'+str(ith_call), do_tight_fit=False)
                    if self.task['deformation - make gif of mesh model driven by the depth map']:
                        self.GifMakerImageDepthMapMesh.add_figure(fig_mesh_edge_change)

                # plot target and mesh based point cloud
                fig_compare_point_clouds, ax = generate_mpl_3D_figure(size_width=3.25, size_height=3.25, do_tight_fit=True)
                fig_compare_point_clouds, ax = plot_single_point_cloud(torch_point_cloud=self.point_cloud_target[:, mask[0, :, 0], :],
                                                  title='compare current shape to target',
                                                  ax=ax, fig=fig_compare_point_clouds, color=red_3, alpha=0.5, thickness=0.03)
                fig_compare_point_clouds, ax = plot_single_point_cloud(torch_point_cloud=self.point_cloud_map.points_packed()[None][:, mask[0, :, 0], :],
                                                  title='compare current shape to target',
                                                  ax=ax, fig=fig_compare_point_clouds, color=green_4, alpha=0.6, thickness=0.05)
                point_cloud1 = self.point_cloud_target[0].cpu().detach().numpy()
                point_cloud2 = self.point_cloud_map.points_packed().cpu().detach().numpy()
                for _, i in enumerate([1, 127, point_cloud2.shape[0]-128, point_cloud2.shape[0]-1]):
                    x = np.array((point_cloud1[i, 0], point_cloud2[i, 0]))
                    y = np.array((point_cloud1[i, 1], point_cloud2[i, 1]))
                    z = np.array((point_cloud1[i, 2], point_cloud2[i, 2]))
                    ax.plot(x, y, z, color=red_2, linestyle='-')
                ax.set_axis_off()
                for ip in [[(0, 0), (0, 90), (90, 0)] if plot_different_perspectives else [(45, 35)]][0]:
                    ax.view_init(elev=ip[0], azim=ip[1])
                    fig_compare_point_clouds.show()
                    ax.title.set_visible(False)
                    save_figure(fig=fig_compare_point_clouds, name_of_figure='deformation depth map compare point clouds iteration_'+str(iteration)+'_th_call_'+str(ith_call)+'_perspective_'+str(ip), do_tight_fit=False)
                if self.task['deformation - make gif of mesh model driven by the depth map']:
                    self.GifMakerImageDepthMapPointCloud.add_figure(fig_compare_point_clouds)

            # calculate loss based on difference of mesh to point cloud
            loss_chamfer, _ = chamfer_distance(x=self.point_cloud_map.points_packed()[None][:, mask[0, :, 0], :], y=self.point_cloud_target[:, mask[0, :, 0], :])
            loss_euclidean = ((self.point_cloud_map.points_packed()[None][:, mask[0, :, 0], :] - self.point_cloud_target[:, mask[0, :, 0], :])**2).mean()

            loss["chamfer distance to points"] = loss_chamfer \
                if self.task['deformation - depth map optimization - chamfer distance'] \
                else torch.tensor(0.0, device=self.device)
            loss["euclidean distance to points"] = loss_euclidean \
                if self.task['deformation - depth map optimization - euclidean distance'] \
                else torch.tensor(0.0, device=self.device)

        else:  # if this loss is not used set loss to 0
            loss["chamfer distance to points"] = torch.tensor(0.0, device=self.device)
            loss["euclidean distance to points"] = torch.tensor(0.0, device=self.device)

        # ----------------------------------------------------------------------
        # Texture optimization
        # ----------------------------------------------------------------------

        # Option for geometry optimization using texture information of special points of interest
        if self.task["deformation - texture optimization"]:  # maybe in future separate of texture opt. above (predict img necesary)

            # Get predicted image from rendering mesh (coarse mesh used here and not the fine one)
            fine_mesh_texture = subdivideMesh(mesh=new_src_mesh, iter=self.task["subdivide level for textured mesh"])
            fine_mesh_texture.textures = self.texture

            images_rendered = self.renderer_texture_observe(fine_mesh_texture,
                                                            cameras=self.depth_map_data.synthetic_camera,
                                                            lights=self.lights_ambient)  # ToDo render a batch >1 at once!
            images_predicted = images_rendered[..., :3]

            # Prepare optimization: measurement image (objetive) as y and predicted image (model) as x
            if iteration == 0:
                with torch.no_grad():
                    def map_features_on_mesh(point_cloud):
                        # Round positions on image pixel grid for previous pc (needed as int)
                        torch.round(point_cloud)
                        img_predicted_pix_pos_of_patterns = point_cloud.type(dtype=torch.long)

                        # Create Map object that links different mesh deformations with the feature point locations
                        texture_points_3d, fragments_map = self.inverse_renderer(meshes_world=new_src_mesh,
                                                                                 # ToDo check if using the fine_mesh would improve anything here
                                                                                 node_pos=img_predicted_pix_pos_of_patterns,
                                                                                 cameras=self.depth_map_data.synthetic_camera,
                                                                                 req_gradient=True, req_fragments=True)
                        map_object = PointCloudMap(texture_points_3d,
                                                   types=None,
                                                   remove_node_type=None,
                                                   fragments_map=fragments_map,
                                                   new_orb_features=None,
                                                   nx_graph=None, device=self.device)
                        return map_object

                    # load hand selected feature points
                    if self.task['deformation - texture optimization - feature pairs']:  # select manually
                        a_points, b_points = self.depth_map_data.image_feature_pairs_positions
                        a_points = np.flip(a_points, axis=1)  # Flip representations of points [x, y] -> [y, x] because
                        b_points = np.flip(b_points, axis=1)  # inverse renderer expect H, W (indices in array)
                        feature_pc_previous, self.feature_pc_targets[0] =\
                            torch.tensor(a_points.copy(), device=self.device),\
                            torch.tensor(b_points.copy(), device=self.device)
                        self.feature_pc_previous_map_object[0] = map_features_on_mesh(feature_pc_previous)

                    if self.task['deformation - texture optimization - skeletons']:  # load extracted  point cloud
                        feature_pc_previous, point_cloud_target = self.depth_map_data.contour_image_pair
                        feature_pc_previous, self.feature_pc_targets[1] = \
                            torch.tensor(feature_pc_previous, device=self.device), \
                            torch.tensor(point_cloud_target, device=self.device)
                        self.feature_pc_previous_map_object[1] = map_features_on_mesh(feature_pc_previous)


            def get_point_cloud_prediction(feature_nr):
                # after every deformation /iteration
                predicted_texture_points_based_on_new_deformed_mesh = self.feature_pc_previous_map_object[feature_nr].\
                    update_point_cloud_based_on_mesh(mesh=new_src_mesh)
                projected_image_keypoints_2d = camera_projection(
                    predicted_texture_points_based_on_new_deformed_mesh.points_packed(),
                    camera=self.depth_map_data.synthetic_camera,
                    image_size=torch.tensor([self.image_size, self.image_size], device=self.device, dtype=torch.float),
                    device=self.device, r_rotation=None, t_translation=None)
                return projected_image_keypoints_2d

            predicted_pix_pos_of_patterns = [None, None]
            if self.task['deformation - texture optimization - feature pairs']:
                target_pix_pos_of_patterns = self.feature_pc_targets[0]
                predicted_pix_pos_of_patterns[0] = get_point_cloud_prediction(0)
                error = ((predicted_pix_pos_of_patterns[0] - target_pix_pos_of_patterns) ** 2).mean()
                loss["texture feature points"] = error
            else:
                loss["texture feature points"] = torch.tensor(0.0, device=self.device)

            if self.task['deformation - texture optimization - skeletons']:
                target_pix_pos_of_patterns = self.feature_pc_targets[1]
                predicted_pix_pos_of_patterns[1] = get_point_cloud_prediction(1)
                error, _ = chamfer_distance(x=predicted_pix_pos_of_patterns[1][None],
                                            y=target_pix_pos_of_patterns[None])
                loss["texture skeleton points"] = error
            else:
                loss["texture skeleton points"] = torch.tensor(0.0, device=self.device)
        else:
            loss["texture features"] = torch.tensor(0.0, device=self.device)
            projected_image_keypoints_2d, y_img_target_pix_pos_of_patterns = None, None
            predicted_pix_pos_of_patterns = [None, None]

        # if needed for texture loss or visualisation: render image of mesh
        if self.task['deformation - texture optimization'] or (
                self.task['deformation - texture optimization'] and iteration % self.task[
                'deformation - depth map optimization - plot period'] == 0):
            fine_mesh_texture = subdivideMesh(mesh=new_src_mesh, iter=self.task["subdivide level for textured mesh"])
            fine_mesh_texture.textures = self.texture

            images_rendered = self.renderer_texture_observe(fine_mesh_texture,
                                                            cameras=self.depth_map_data.synthetic_camera,
                                                            lights=self.lights_ambient)  # ToDo render a batch >1 at once!
            images_predicted = images_rendered[..., :3]

        # Option for geometry optimization using general texture information
        if self.task['deformation - texture optimization - rgb data']:
            # Squared L2 distance between the predicted RGB image and the target
            visual_image_loss = (images_predicted-self.depth_map_data.torch_measurement_image)**2
            loss["image rgb data"] = visual_image_loss.mean()
        else:  # if this loss is not used set loss to 0
            loss["image rgb data"] = torch.tensor(0.0, device=self.device)

        # visualize texture if one of the texture optimization algorithms was used
        if self.task['deformation - texture optimization'] and iteration % self.task['deformation - depth map optimization - plot period'] == 0:
            # only for test \/
            images_rendered = self.RenderersCollection.renderer_view(fine_mesh_texture,
                                                                     # cameras=self.depth_map_data.synthetic_camera,
                                                                     # lights=self.lights_ambient
                                                                     )  # ToDo render a batch >1 at once!
            images_predicted = images_rendered[..., :3]
            # only for test /\
            fig, axs = visualize_features_on_image(images_predicted.detach(),
                                                   self.depth_map_data.torch_measurement_image.detach()[None,...],
                                                   title='Comparison of predicted an target image',
                                                   predictions=predicted_pix_pos_of_patterns,
                                                   targets=self.feature_pc_targets,
                                                   )
            save_figure(fig, f'Comparison of real image and recreation through '
                             f'model after deformation iteration_{iteration}',
                        show_fig=True)
            if self.task['deformation - make gif of mesh model driven by the depth map']:
                self.GifMakerImageDepthMapTexture.add_figure(fig)

        # ----------------------------------------------------------------------
        # Loss
        # ----------------------------------------------------------------------

        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=self.device)
        for k, l in loss.items():
            weighted_loss = l * self.losses[k]["weight"]
            sum_loss += weighted_loss
            with torch.no_grad():
                self.losses[k]["values"].append(float(weighted_loss.detach().cpu()))


        # delete graph of mesh in a way that it can be used in further optimizations
        deform_verts_param_no_grad = self.deform_verts_param.clone().detach()

        deform_verts_param_no_grad.requires_grad = False
        return_src_mesh = self.src_mesh.offset_verts(deform_verts_param_no_grad)
        return sum_loss, return_src_mesh

    def align_point_cloud_on_mesh(self, mesh, node_pos_key, point_cloud_keys_2b_aligned,
                                  R_measurement_pc_init=None, T_measurement_pc_init=None, #optional
                                  overall_point_clouds=None, losses_pc_alignment=None):
        point_cloud_key_from_mesh, _ = self.inverse_renderer(meshes_world=mesh,
                                                             node_pos=node_pos_key,
                                                             req_gradient=False,  # set to False
                                                             req_fragments=True,
                                                             R=self.camera_model_view.R, T=self.camera_model_view.T)


        from models.pc_alignment_pose_opt import PointCloudAlignmentOptimization

        model_point_cloud_alignment = PointCloudAlignmentOptimization(device=self.device, losses=losses_pc_alignment, camera_view_model_synthetic=self.camera_model_view)
        r_rotation_pc, t_translation_pc, scale_xy_plane = \
            model_point_cloud_alignment(point_cloud_with_desired_pose=point_cloud_key_from_mesh.points_packed()[None],
                                        point_cloud_keys_2b_aligned=point_cloud_keys_2b_aligned,
                                        R_measurement_pc_init=R_measurement_pc_init, T_measurement_pc_init=T_measurement_pc_init,
                                        iterations=self.task['deformation - depth map optimization - number of iterations for initial pose alignment'],
                                        lr=self.task['deformation - depth map optimization - learning rate for initial pose alignment'],
                                        overall_point_clouds=overall_point_clouds,
                                        mesh_for_plotting = mesh)


        # optimization problem is sensitive with respect to the learning rate!
        print('Scaling parameter xy-plane :', model_point_cloud_alignment.get_scale_xy_plane)
        R_pc = self.cam_rodrigues_object.get_rotation_matrix(r_rotation_pc)
        T_pc = self.cam_rodrigues_object.get_translation_matrix(t_translation_pc)

        if PLOT_KEY_POINT_CLOUD_ON_MESH:
            from utils.world_map_visualization import plot_mesh
            fig, ax = plot_mesh(mesh_model=mesh)

            point_cloud_target = point_cloud_key_from_mesh.points_packed().cpu().detach().numpy()
            ax.scatter(xs=point_cloud_target[:, 0], ys=point_cloud_target[:, 1], zs=point_cloud_target[:, 2],
                       color=red_2, alpha=0.7, s=14)
            fig.show()


        return R_pc, T_pc, scale_xy_plane

    def define_point_cloud(self, mesh, R, T, image_dim):

        if self._all_image_node_pos is None:
            dim_in_x = image_dim[1]
            dim_in_y = image_dim[0]
            self._all_image_node_pos = torch.zeros((dim_in_x*dim_in_y, 2), dtype=torch.long)
            for x_column_itr in range(dim_in_x):
                # in image coordinate system x corresponds to columns and y- to rows!!
                self._all_image_node_pos[range(x_column_itr * dim_in_y, (x_column_itr + 1) * dim_in_y), :] = torch.cat((
                                                        torch.full((dim_in_y,), x_column_itr, dtype=torch.long)[:, None],
                                                        torch.tensor(range(0, dim_in_y), dtype=torch.long)[:, None]
                                                        ), 1)

        point_cloud, point_cloud_fragments_map = self.inverse_renderer(meshes_world=mesh,
                                                                       node_pos=self._all_image_node_pos,
                                                                       req_gradient=False, #set to False
                                                                       req_fragments=True,
                                                                       R=R, T=T)

        point_cloud_map = PointCloudMap(PointCloud=point_cloud,
                                        fragments_map=point_cloud_fragments_map,
                                        device=self.device)
        if PLOT_RE_RENDERED_KEY_POINT_CLOUD_ON_MESH:
            from tools_generate.DataDepthMap import plot_torch_3dtensor
            plot_torch_3dtensor(torch_point_cloud_tensor=point_cloud_map.points_packed()[None])

        return point_cloud_map


    def visualize_mesh(self, title='', batch_size=None, if_save=True, make_gif=False):
        print('Function visualize_mesh() to be implemented')

    def visualize_texture_prediction(self, title=''):
        print('Function visualize_texture_prediction() to be implemented')



    @property
    def get_loss_dict_history(self):
        return self.losses

    @property
    def get_pc_alignment_transformation(self):
        return self.R_pc_alignment, self.T_pc_alignment, self.scale_xy_plane_alignment


    def close_existing_gifs(self):
        if self.task['deformation - make gif of mesh model driven by the depth map']:
            self.GifMakerImageDepthMapMesh.close_writer()
            self.GifMakerImageDepthMapPointCloud.close_writer()



def mesh_edge_outlier_loss(meshes, k=2):
    # Todo: Work in progress
    """
    Computes a value representing how much the length of the edges in the mesh
    differs from the mean edge length
    Args:
        meshes: Meshes object with a batch of meshes.
    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)

    lengths = torch.norm((v0 - v1), dim=1)
    std_length, mean_length = torch.std_mean(lengths, dim=0, unbiased=True)

    threshold = mean_length + k*std_length
    lengths_with_threshold = lengths - threshold
    #diff_lengths = lengths - mean_length
    #loss = diff_lenghts.topk(k)[0].mean() - diff_lenghts.topk(k, largest=False)[0].mean()
    sigmoid_function = torch.nn.Sigmoid()
    loss = sigmoid_function(lengths_with_threshold)

    return loss.sum()

def update_mesh_shape_prior_losses(mesh, loss, edge_consistency_loss=False, factor=1):
    # looks for smoothness, so for a perfect sphere the loss is zero and the loss goes up later
    # Losses to smooth / regularize the mesh shape

    # and (b) the edge length of the predicted mesh
    loss["edge"] = factor*mesh_consistency_edge_loss(mesh) if edge_consistency_loss \
        else factor*mesh_edge_loss(mesh)

    # mesh normal consistency
    loss["normal"] = factor*mesh_normal_consistency(mesh)

    # mesh laplacian smoothing
    loss["laplacian"] = factor*mesh_laplacian_smoothing(mesh, method="uniform")

    return loss


def visualize_image_deformation_with_target(predicted_images,
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
    #Todo return fig

def plot_deformation_error(group_of_nodes:dict,target_nodes_np, predicted_nodes_np,image_size ):
    #fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    fig, ax = plt.subplots()
    for j in range(target_nodes_np.shape[0]):
        x = np.array((predicted_nodes_np[j][0], target_nodes_np[j][0]))
        y = np.array((image_size - predicted_nodes_np[j][1], image_size - target_nodes_np[j][1]))
        # Plot the connecting lines
        ax.plot(x, y, color= red_2, linestyle='-')
    for i in range(group_of_nodes.get('data').__len__()):
        scale = 40
        ax.scatter(x=group_of_nodes.get('data')[i][0],
                   y=group_of_nodes.get('data')[i][1],
                   s=scale,
                   color=group_of_nodes.get('color')[i],
                   label=group_of_nodes.get('labels')[i],
                   alpha=group_of_nodes.get('alpha')[i],
                   marker=group_of_nodes.get('marker')[i],
                   edgecolors='none')
    plt.title('Ratio between target and updated node position.')
    # Turn off tick labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.axis('off')
    ax.legend()
    scale_for_legend = 1.3
    plt.xlim(0, image_size * scale_for_legend)
    plt.ylim(0, image_size)
    plt.show()
    return fig


def plot_deformation_on_graph(group_of_nodes: dict, target_nodes_np, predicted_nodes_np, adjacency: np.ndarray = None, nx_graph = None, image_size=128, image_rgb=None, plot_graph=True, title:str=None):

    fig, ax = plt.subplots()
    if image_rgb is not None:
        y_lim = image_size
        x_lim = image_size
        extent = 0, x_lim, 0, y_lim
        ax.imshow(image_rgb, extent=extent, interpolation='nearest')
    if plot_graph:
        if adjacency is not None:
            adjacency_matrix = np.uint8(adjacency.copy())
            Graph = nx.from_numpy_matrix(adjacency_matrix)
        else:
            Graph = nx_graph

        # set positions to target or predicted noe positions!!
        positions = predicted_nodes_np
        pos_list = []
        for i in range(len(positions)):
            pos_list.append(
                [positions[i][0],
                 image_size - positions[i][1]])  # flip y-axis, since node_pos are in image coordinates
        p = dict(enumerate(pos_list, 0))
        nx.set_node_attributes(Graph, p, 'pos')
        nx.draw(Graph, pos=p, node_size=0, edge_color=green_4, width=1, node_color=None)


    for j in range(target_nodes_np.shape[0]):
        x = np.array((predicted_nodes_np[j][0], target_nodes_np[j][0]))
        y = np.array((image_size - predicted_nodes_np[j][1], image_size - target_nodes_np[j][1]))
        # Plot the connecting lines
        ax.plot(x, y, color=red_2, linestyle='-')

    for i in range(group_of_nodes.get('data').__len__()):
        scale = 40
        ax.scatter(x=group_of_nodes.get('data')[i][0],
                   y=group_of_nodes.get('data')[i][1],
                   s=scale,
                   color=group_of_nodes.get('color')[i],
                   label=group_of_nodes.get('labels')[i],
                   alpha=group_of_nodes.get('alpha')[i],
                   marker=group_of_nodes.get('marker')[i],
                   edgecolors='none')
    if title is not None:
        plt.title(title)
    else:
        plt.title('Ratio between target and updated node position ' + "\n" + 'along extracted graph pattern.')
    # Turn off tick labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.axis('off')
    #ax.legend()
    scale_for_legend = 1
    plt.xlim(0, image_size * scale_for_legend)
    plt.ylim(0, image_size)
    plt.show()
    return fig


def plot_point_clouds(torch_point_cloud_current, torch_point_cloud_target  , title:str ='', ax=None, fig=None):
    if torch_point_cloud_current.dim() == 3:
        point_cloud_current = torch_point_cloud_current[0].cpu().detach().numpy()
        point_cloud_target = torch_point_cloud_target[0].cpu().detach().numpy()
    else: print('Incorrect shape!')
    with plt.style.context(('ggplot')):
        if ax is None or fig is None:
            fig = plt.figure(figsize=(10, 7))
            ax = Axes3D(fig)
        ax.scatter(xs=point_cloud_current[:, 0], ys=point_cloud_current[:, 1], zs=point_cloud_current[:, 2],
                       color=red_2, alpha=0.7, s=1)
        ax.scatter(xs=point_cloud_target[:, 0], ys=point_cloud_target[:, 1], zs=point_cloud_target[:, 2],
                       color=green_4, alpha=0.7, s=1)
        ax.set_title('Scatter plot of pytorch tensor'+title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        fig.show()
        return fig, ax


def plot_single_point_cloud(torch_point_cloud, title:str ='', ax=None, fig=None, alpha=0.7, thickness=1, color=green_4):
    if torch_point_cloud.dim() == 3:
        point_cloud = torch_point_cloud[0].cpu().detach().numpy()
    else: print('Incorrect shape!')
    with plt.style.context(('ggplot')):
        if ax is None or fig is None:
            fig, ax = generate_mpl_3D_figure()
        ax.scatter(xs=point_cloud[:, 0], ys=point_cloud[:, 1], zs=point_cloud[:, 2],
                       color=color, alpha=alpha, s=thickness)
        ax.set_title('Scatter plot of pytorch tensor'+title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        #fig.show()
        return fig, ax


def visualize_numpy_images(image, scnd_image=None,
                           title='', scnd_title='',
                           size_width=3.25, size_height=3.25):
    # using the variable axs for multiple Axes
    plt = get_plt_style()
    plt.title(title)
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    fig = size_fig(fig, size_width, size_height)

    axs[0].imshow(image)
    axs[0].title.set_text(title)
    axs[0].set_axis_off()
    if scnd_image is not None:
        axs[1].imshow(scnd_image)
        axs[1].title.set_text(scnd_title)
        axs[1].set_axis_off()
    fig.tight_layout()
    return fig, axs


def visualize_features_on_image(image1, image2, predictions=(), targets=(),
                                node_thickness=15, title='Features on Image', size_width=7.5, size_height=7.5):
    """
    Args:
        image1: np array of image rendered of model
        image2: np array of target image
        predictions: iterable containing point cloud (format H,W) in predicted position
        targets: iterable containing point cloud (format H,W) in target position
    Returns: fig, axs
    """

    plt = get_plt_style()
    plt.title(title)
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    fig = size_fig(fig, size_width, size_height)

    for image, i in zip([image1, image2], range(2)):
        if torch.is_tensor(image):
            img = image.detach().cpu().numpy()[0]
        else:
            img = image  # needs to be a numpy array!
        # Plot Image in image axis style (origin upper left)
        axs[i].imshow(img, origin='upper', extent=(0, img.shape[1], img.shape[0], 0))

    for prediction, target in zip(predictions, targets):
        if torch.is_tensor(prediction):
            prediction = prediction.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()

        if prediction is not None and target is not None and len(prediction) == len(target):  # assume pair if len equal
            for p1, p2 in zip(prediction, target):
                axs[0].plot((p1[1], p2[1]), (p1[0], p2[0]), linewidth=node_thickness/8)
            axs[0].scatter(x=prediction[:, 1], y=prediction[:, 0], marker='x', s=node_thickness, c=blue_2, alpha=0.9)
            axs[0].scatter(x=target[:, 1], y=target[:, 0], marker='+', s=node_thickness, c=red_1, alpha=0.9)
            axs[1].scatter(x=target[:, 1], y=target[:, 0], marker='x', s=node_thickness, c=red_1, alpha=0.9)
        else:  # assume point cloud (sceleton or similar)
            if prediction is not None:  # positions in order H, W like they are used in inverse renderer/ renderer
                axs[0].scatter(x=prediction[:, 1], y=prediction[:, 0], s=node_thickness/4, c=blue_2, alpha=0.5)
            if target is not None:
                axs[0].scatter(x=target[:, 1], y=target[:, 0], s=node_thickness/4, c=red_1, alpha=0.9)
                axs[1].scatter(x=target[:, 1], y=target[:, 0], s=node_thickness/4, c=red_1, alpha=0.9)

    for i in range(2):
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')
        fig.tight_layout()
    #fig.show()
    return fig, axs


####
def is_any_nan(torch_tensor):
    is_nan = torch.any(torch.isnan(torch_tensor))
    print('Tensor has nan values! :', is_nan)