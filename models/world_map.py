import numpy as np
import torch
import torch.nn as nn
# rendering components
from pytorch3d.renderer import (
    PerspectiveCameras
)
from models.deformation import Deformation
import copy
from models.point_cloud_features import PointCloudMap
from tools_graph.match_orb_tools import match
from tools_graph.plot_functs import plot_graph_matches_color
from models.ransac import triangulate_ransac, calibrate_intrinsic_camera_parameters
from models.pose_rodrigues_rot_fromula import Camera3DPoseRodriguesRotFormula
import networkx as nx
from utils.camera_visualization import plot_cameras
from models.camera_pose import camera_projection
from tools_generate.node_classifiers import NodeTypes
from configs.plot.colours import *
from configs.plot.config_plots import *
from configs.config import *
from models.camera_pose_opt import CameraPoseOpt
from utils.world_map_visualization import create_world_map_mesh_and_graph_figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tools_generate.UpComingDataGenerator import UpComingDataGenerator
from utils.mesh_operations import subdivideMesh
from models.texture import TextureTypes
from models.texture import Texture
from tools_generate.distortion import distort_image
from utils.deformation_helpers import NodePointContainer
from tools_generate.image_processing import skeletonise_and_clean
from utils.deformation_helpers import rotateImage
from utils.plot_images import visualize_image, visualize_difference_of_grey_scale_images_by_rgb
from utils.world_map_visualization import plot_mesh
from models.texture import InitRenderers

class WorldMap(nn.Module):

    def __init__(self, data_generator, mesh_model, device, RenderersCollection=None, camera_init=None, renderer_settings=None, r_rotation: torch.tensor=None, task_deformation_update_distortion=None,
                 t_translation: torch.tensor=None,
                 task=None,
                 print_loss=False,
                 calibrate_cameras=True,
                 losses_camera_pose=None):
        super().__init__()
        self.seen_images = list()
        self.seen_images_of_intrest = list()
        self.device = device

        self.calibrate_cameras = calibrate_cameras

        if isinstance(RenderersCollection, InitRenderers):
            self.RenderersCollection = RenderersCollection
        else:
            self.RenderersCollection = InitRenderers(camera_texture=camera_init.clone(), camera_view=None,
                                                     renderer_settings=renderer_settings, device=device)

        self.image_size = torch.tensor([self.RenderersCollection.image_size, self.RenderersCollection.image_size],
                                       device=device)  # used in ransac model

        self.inverse_renderer = self.RenderersCollection.inverse_renderer
        #self.inverse_renderer = inverse_renderer

        self.camera_init = camera_init.clone()  # used in ransac model
        self.camera = camera_init.clone()

        self.camera_view_only = PerspectiveCameras(focal_length=0.5 * self.camera.focal_length, device=device)

        # Pose Parameter:
        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=self.device)
        self.r_rotation_latest = self.cam_rodrigues_object.get_rot_vec_from_rotation_matrix(R_matrix=self.camera_init.R)
        self.t_translation_latest = self.cam_rodrigues_object.get_translation_matrix(T_absolute=self.camera_init.T)
        self.r_rotation_init = self.r_rotation_latest.clone()
        self.t_translation_init = self.t_translation_latest.clone()

        # # Pose Parameter:
        # self.r_rotation_init = r_rotation.to(self.device).clone()
        # self.t_translation_init = t_translation.to(self.device).clone()
        # self.r_rotation_latest = r_rotation.to(self.device).clone()
        # self.t_translation_latest = t_translation.to(self.device).clone()

        # R_init, T_init = self.cam_rodrigues_object.get_rotation_and_translation(self.r_rotation_init,
        #                                                                         self.t_translation_init)
        # self.R_init = R_init.clone()
        # self.T_init = T_init.clone()
        self.pos_init = None
        self.mesh_model = mesh_model
        self.data_generator = data_generator
        self.init_graph_image = None
        self.current_graph_image = None
        self.current_image_rgb = None
        self.current_image_grey = None
        self.current_point_cloud = None
        self.R_latest = None
        self.T_latest = None
        self.print_loss = None
        self.__model_camera_pose = None
        self.__optimizer = None
        self.model_camera_pose = None
        self.iterations = 500
        self.print_loss = print_loss
        self.point_cloud_init = None
        self.world_point_cloud_map = None
        self.current_batch_object = None
        self.currently_loaded_batch_nr = None  # do not touch that variable!! It tracks the currently loaded batch in order to avoid reloading
        self.currently_loaded_batch_item = None  # do not touch that variable!! It tracks the currently loaded batch in order to avoid reloading
        self.desired_item_in_batch_ofcurrentinterest = 0
        self.desired_batch_nr_ofcurrentinterest = 0
        self.world_map_matches = list()
        self.remove_node_type = NodeTypes.BORDER

        self.losses_camera_pose = {"euclidean": {"weight": 1.0, "values": []},
                           "normal": {"weight": 0.8, "values": []},
                           "pull-back": {"weight": 0.0, "values": []}
                           }
        if losses_camera_pose is None:
            self.losses_camera_pose = {"euclidean": {"weight": 150, "values": []},
                               "normal": {"weight": 100, "values": []},
                               }
        else:
            self.losses_camera_pose = losses_camera_pose

        # self.losses_camera_pose_rgb = {
        #     "rgb": {"weight": 0.5, "values": []}}
        deform_loss = {
            "texture chamfer loss": {"weight": 3.8, "values": []},
            "pure texture": {"weight": 0.0, "values": []},
            "edge": {"weight": 1.1, "values": []},
            "normal": {"weight": 1.8, "values": []},
            "laplacian": {"weight": 1.2, "values": []},
        }

        self.task = {
                     'Graph based Feature Assignment - for 3d matching and reconstructions': True,
                     "init texture usage": False,
                     "texture based deformation": False,
                     "texture based deformation -test with distortion": False,
                     "texture based pose optimization": False,
                     "deformation - use remaining error for deformation": False,
                     "texture learning": True,
                     "pose by texture": False,
                     "fix first pose": True,
                     "use chamfer loss": False,
                     "batch size for rendering": 5,
                     "number of last images 2b used": 10,
                     "learning rate - texture": 1e-1,
                     "learning rate - pose": 1e-1,
                    '3D RANSAC number of RANSAC samples that are tested': 35,
                    '3D RANSAC number of iterations per sample': 41,
                    '3D RANSAC reporjection error epsilon': 0.5,
                    'Texture based pose opt - number of iterations': 35,
                    'Texture Learning number of iterations': 44,
                     "deformation - make 3D mesh plot after calling remaining error in world_map": False,
                     'Plot init Graph on 3D Mesh': False,
                     'Make Gif and plot graph on 3D Mesh in .forward() call': False,
                     'UpComingDataGenerator plot matches between images - for final sample': False,
                     'UpComingDataGenerator matches and make Gif - for all upcoming samples': False,
                     'UpComingDataGenerator number of minimal matches': 65,
                     'Plot matches between current and newly loaded graph image - world_map':False,
                     'Texture Reconstruction - load input data from seen_images else function parameters have to be set!':True,
            'texture based deformation -test with distortion - Parameter: deform_loss': deform_loss,
        }
        if task is not None:
            self.task.update(task)

        self.task_deformation_update_distortion = task_deformation_update_distortion

        self.ModelCameraPoseOpt = CameraPoseOpt(inverse_renderer=self.RenderersCollection.inverse_renderer_nodePoseOpt, losses=self.losses_camera_pose)
        self.UpComingDataGenerator = UpComingDataGenerator(data_generator=data_generator, device=device)

        #self.init()
        ## Model Parameter:
        self.TextureModelVisualization = None
        self.renderer_settings = self.RenderersCollection.renderer_settings

        self.init_load_graph_img()
        if self.task['Graph based Feature Assignment - for 3d matching and reconstructions']:
            self.init_graph_on_3D_mesh()

        if  self.task["init texture usage"]:
            self.init_texture_usage()


    def init_texture_usage(self):
            ## texture optimization
            fine_mesh_texture = subdivideMesh(mesh=self.mesh_model.clone(), iter=self.task["subdivide level for textured mesh"])

            losses_texture =  {"chamfer single iteration loss": {"weight": 1.0, "values": []},
                               "chamfer every iteration loss": {"weight": 1.0, "values": []},
                               "image pattern loss": {"weight": 1.0, "values": []},
                               "texture reconstruction loss": {"weight": 1.0, "values": []}}

            render_settings = {"face per pixel - opt": 25, #20 #Still makes sense to not set it to zero, since it covers more space for fine meshes using course images!
                                  "blur radius - opt": 1*1e-6,
                                  "blend param sigma - opt": 1e-7,
                                  "blend param gamma - opt": 1*1e-5,
                                  "face per pixel - view": 1,
                                  "blur radius - view": 1e-6,
                                  "blend param sigma - view": None,
                                  "blend param gamma - view": None,
                                  "face per pixel - inverse": 1,
                                  "blur radius - inverse": np.log(1. / 1e-4 - 1.) * 1e-5,
                                  "blend param sigma - inverse": 1e-5,
                                  "blend param gamma - inverse": 1e-4,
                                  "texture type for reconstruction": TextureTypes.VERTEX_TEXTURE_GREY,}

            self.renderer_settings.update(render_settings)

            self.TextureModel = Texture(mesh=fine_mesh_texture, camera_texture=self.get_init_camera_perspective(),
                                        losses=losses_texture, RenderersCollection= self.RenderersCollection, renderer_settings=self.renderer_settings,
                                        UpComingDataGenerator = self.UpComingDataGenerator, device=self.device, geometry_mesh=self.mesh_model, task=self.task)
            #mask = generate_mask(node_pos=self.current_graph_image.node_pos, mask_size=(self.current_graph_image.image_size, self.current_graph_image.image_size), return_as_torch=True, device=self.device)
            if self.renderer_settings["texture type for reconstruction"] == TextureTypes.VERTEX_TEXTURE_RGB or self.renderer_settings["texture type for reconstruction"] == TextureTypes.VERTEX_TEXTURE:
                self.TextureModel.forward(target_images=torch.tensor(self.current_image_rgb/255, device=self.device)[None], image_mask=self.current_graph_image.generate_mask_for_image(return_as_torch=True), target_cameras=self.get_current_camera_position,
                                      task=self.task, iterations=self.task['Texture Learning number of iterations'], lr=5*1e-2)
            elif self.renderer_settings["texture type for reconstruction"] == TextureTypes.VERTEX_TEXTURE_GREY:
                self.TextureModel.forward(target_images=torch.tensor(self.current_image_grey/255, device=self.device)[None], image_mask=self.current_graph_image.generate_mask_for_image(return_as_torch=True), target_cameras=self.get_current_camera_position,
                                      task=self.task, iterations=self.task['Texture Learning number of iterations'], lr=5*1e-2)

            if self.task["texture based deformation"]:

                # render_settings = {"face per pixel - opt": 25,
                #                    "blur radius - opt": 1e-4,
                #                    "blend param sigma - opt":  1 * 1e-6,
                #                    "blend param gamma - opt": 2 * 1e-4,
                #                    "face per pixel - view": 1,
                #                    "blur radius - view": 1e-6,
                #                    "blend param sigma - view": None,
                #                    "blend param gamma - view": None,
                #                    "face per pixel - inverse": 25,
                #                    "blur radius - inverse": np.log(1. / 1e-4 - 1.) * 1e-4,
                #                    "blend param sigma - inverse": 1e-7,
                #                    "blend param gamma - inverse": 1*1e-4}
                # self.renderer_settings.update(render_settings)
                render_settings = {"face per pixel - opt": 2,
                                   "blur radius - opt": 1e-9,
                                   "blend param sigma - opt": 1e-9,
                                   "blend param gamma - opt": 2 * 1e-8,


                                   "face per pixel - high opt": 40,
                                   "blur radius - high opt": np.log(1. / 1e-4 - 1.) * 1e-2,
                                   "blend param sigma - high opt": 1 * 1e-3,
                                   "blend param gamma - high opt": 4 * 1e-1,

                                   "face per pixel - view": 1,
                                   "blur radius - view": 1e-6,
                                   "blend param sigma - view": None,
                                   "blend param gamma - view": None,
                                   "face per pixel - inverse": 4,
                                   "blur radius - inverse": 1 * np.log(1. / 1e-4 - 1.) * 1e-5,
                                   "blend param sigma - inverse": 1e-5,
                                   "blend param gamma - inverse": 1*1e-5}
                #test to decrease inverse and increase opt rendering blurring!

                self.renderer_settings.update(render_settings)
                self.RenderersCollection.update_renderer(renderer_settings=render_settings)
                #self.TextureModel.get_RenderersCollection.update_renderer(renderer_settings=self.renderer_settings)
                self.DeformationModel = Deformation(world_map=self,
                                                    geometry_mesh=None,
                                                    losses=None,
                                                    camera=self.get_init_camera_perspective(),
                                                    RenderersCollection=self.RenderersCollection,
                                                    UpComingDataGenerator=self.UpComingDataGenerator,
                                                    depth_map_dim=(128, 128),
                                                    task=self.task,
                                                    )
                if self.task["texture based deformation -test with distortion"]:




                    def get_camera_params():
                        h = 128
                        w = 128
                        # camera parameters: ret, mtx, dist, rvecs, tvecs
                        fx = 1.1  # 1058 #ToDo f is not adapting node postions
                        fy = 1.2  # 1041
                        cx = w / 3
                        cy = h / 2

                        fx = 1.0# 1058 #ToDo f is not adapting node postions
                        fy = 0.7 # 1041
                        cx = w / 3
                        cy = h / 2

                        # # distortion parameters
                        # k1 = 0.00000025
                        # k2 = 0.000000025
                        # p1 = -0.0025
                        # p2 = -0.0015

                        # distortion parameters
                        k1 = 0.0000615
                        k2 = 0.0
                        p1 = -0.0000425
                        p2 = 0.0000615
                        # convert for opencv
                        mtx = np.matrix([
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]
                        ], dtype="float32")

                        dist = np.array([k1, k2, p1, p2], dtype="float32")
                        cameraparams = [0, mtx, dist]
                        return cameraparams


                    undist_pos, image_grey_distorted2 = distort_image(cameraparams=get_camera_params(), image=self.current_image_grey,
                                                             input_pos=self.current_graph_image.get_all_image_nodes, plot_it=True)

                    rotated_image = rotateImage(image_grey_distorted2, angle=-5)

                    def visualize_image(img, size_width=3.75, size_height=3.75, title=''):
                        fig, ax = generate_mpl_figure(size_width=size_width, size_height=size_height)
                        #ax.imshow(img, cmap = 'gray', vmin = 0, vmax = 1.0)
                        ax.imshow(img)
                        ax.axis('on')
                        ax.set_title(title)
                        fig.tight_layout()
                        # fig.show()
                        return fig, ax

                    fig, ax = visualize_image(rotated_image, size_width=3.75, size_height=3.75, title='')
                    fig.show()

                    #################################### #ToDo check!
                    image_grey_distorted = rotated_image*255
                    skeletonized_image_grey = skeletonise_and_clean(
                        thr_image=image_grey_distorted, plot=True, save=False, directory='')

                    fig, ax = visualize_difference_of_grey_scale_images_by_rgb(img_original=self.current_image_grey, img_desired = skeletonized_image_grey, size_width=3.75, size_height=3.75, title='Distored Image vs Target Image')
                    fig.show()

                    # fig, ax = visualize_image(skeletonized_image_grey, size_width=3.75, size_height=3.75, title='')
                    # fig.show()
                    image_grey_distorted = skeletonized_image_grey
                    #######################################

                    data_rotation_translation_batch_lists = [self.current_graph_image.get_rotation_vector, self.current_graph_image.get_translation_vector]

                    data_target_imag_batch = torch.from_numpy(image_grey_distorted)[None].to(device=self.device)
                    #undist_pos = self.current_graph_image.get_all_image_nodes
                    #data_target_imag_batch = torch.from_numpy(self.current_image_grey)[None].to(device=self.device)

                    undist_pos, mask_for_image_distorted = distort_image(cameraparams=get_camera_params(),
                                                             image=self.current_graph_image.generate_mask_for_image(),
                                                             input_pos=self.current_graph_image.get_all_image_nodes, plot_it=True)

                    undist_pos, matching_mask_for_image_distorted = distort_image(cameraparams=get_camera_params(),
                                                             image=self.current_graph_image.generate_matching_mask_for_image(),
                                                             input_pos=self.current_graph_image.get_all_image_nodes, plot_it=True)
                    data_mask_batch_lists = [torch.from_numpy(mask_for_image_distorted)[None].to(device=self.device).type(dtype=torch.bool),
                                             torch.from_numpy(matching_mask_for_image_distorted)[None].to(device=self.device).type(dtype=torch.bool)]

                    task = {"deformation laod data from seen_image list": False,
                            "deformation optimizer SGD": False,
                            "deformation optimizer AdamW": True,
                            "deformation optimizer learning rate": 15*1e-4, #13*1e-4,AdamW
                            "deformation optimizer momentum SGD": 0.9,
                            "deformation optimizer number of iterations": 102,

                            "deformation use node point landmarks on world map": False, #ToDo test with landmarks!
                            "deformation use world map node points for masking chamfer pcs": True,
                            "deformation - use Deformation Model Texture": True,
                            "deformation skeletonize predicted image before point cloud extraction": False,
                            "deformation use chamfer loss": True,
                            "deformation use chamfer loss in 3D point cloud": False, # just 2D or 3D chamfer loss can be used!
                            "deformation use chamfer loss in 2D in image plane": True,
                            "deformation use texture comparison": True, # then also "deformation use chamfer loss" has to be true!
                            "deformation threshold for point cloud extraction": 0.43,
                            "deformation batch size of view for rendering deformed mesh": 12,

                            "deformation use chamfer loss - make plots": True,
                            "use chamfer loss - make plots - all variations in period of": 20,
                            "deformation use chamfer loss - make plots - all variations": False,
                            "number of last images 2b used for deformation": 1,
                            "deformation batch size for rendering": 1,
                            "texture based chamfer optimization": False,
                            "deformation optim - graph nodes": False,
                            "deformation optim - 3d target point cloud": False,
                            }
                    self.task.update(task)


                    deform_loss_new = self.task['texture based deformation -test with distortion - Parameter: deform_loss']

                    if self.task["deformation use node point landmarks on world map"] or self.task["deformation use world map node points for masking chamfer pcs"]:
                        node_point_container_extern = NodePointContainer(world_point_cloud_map=self.world_point_cloud_map,
                                                                       matches_with_world_map_list=[self.current_graph_image.get_matches_to_world_map],
                                                                       points_2D_map_list=[undist_pos],
                                                                       split_indices_crossing_list=[self.current_graph_image.get_crossing_node_indices_by_removed_border],
                                                                       split_indices_border_list=[self.current_graph_image.get_end_node_indices_by_removed_border])
                    else:
                        node_point_container_extern = None
                    task_deformation_update2 = {
                        "deformation - make 3D mesh plot after calling remaining error in world_map": False}
                    if self.task_deformation_update_distortion is not None:
                        task_deformation_update2.update(self.task_deformation_update_distortion)

                    self.deform_remaining_texture_error(data_rotation_translation_batch_lists = data_rotation_translation_batch_lists,
                                                         data_target_imag_batch = data_target_imag_batch, data_mask_batch_lists = data_mask_batch_lists, deform_loss_new=deform_loss_new,
                                                        node_point_container_extern=node_point_container_extern, task_deformation_update= task_deformation_update2)

    def init_load_graph_img(self):
        self.graph_image_init, image_rgb, image_grey = self.UpComingDataGenerator.load_and_create_graph_image_object(
            desired_batch_nr=self.desired_batch_nr_ofcurrentinterest,
            item_in_batch=self.desired_item_in_batch_ofcurrentinterest,
            r_rotation=self.r_rotation_init, t_translation=self.t_translation_init)
        self.graph_image_init.set_remove_node_type = self.remove_node_type
        self.current_image_rgb = image_rgb
        self.current_image_grey = image_grey
        # self.R_latest = self.R_init
        # self.T_latest = self.T_init
        self.current_graph_image = self.graph_image_init
        self.pos_init = copy.deepcopy(self.graph_image_init.node_pos)

    def init_graph_on_3D_mesh(self):

        self.point_cloud_init, fragments_map = self.inverse_renderer(meshes_world=self.mesh_model,
                                                                     node_pos=self.pos_init,
                                                                     req_gradient=True,
                                                                     req_fragments=True,
                                                                     camera = self.camera_init,
                                                                     )


        # self.point_cloud_init, fragments_map = self.inverse_renderer(meshes_world=self.mesh_model,
        #                                                              node_pos=self.pos_init,
        #                                                              req_gradient=True,
        #                                                              req_fragments=True,
        #                                                              camera = self.camera,
        #                                                              R=self.R_init,
        #                                                              T=self.T_init)

        # admissible matches need to be between the desired node types in the world map
        # and all nodes in the current image!
        # first all nodes of all given type
        # if we add new points (of all tyes) we remove the old nodes of type border
        # if we see a match with a node of type end we could update the position with a better replacement!
        init_match_graph = torch.arange(0, self.current_graph_image.nx_graph.number_of_nodes(), dtype=torch.long)[self.current_graph_image.get_crossing_and_end_nodes_indices]
        init_match_world = torch.arange(0, init_match_graph.shape[0], dtype=torch.long)
        init_match = torch.vstack((init_match_world, init_match_graph)).t()
        mask_nan = torch.logical_not(torch.isnan(self.point_cloud_init.points_packed()[self.current_graph_image.get_crossing_and_end_nodes_indices]))[:,
                   0].cpu().detach()
        self.current_graph_image.set_matches_with_world_map(init_match[mask_nan])

        # prepend list by using insert(0, 1) -> latest entry is at first position!
        self.seen_images.insert(0, [self.desired_batch_nr_ofcurrentinterest, self.desired_item_in_batch_ofcurrentinterest, True])
        self._add_current_graph_image_to_seen_images(self.current_graph_image)
        new_graph = self.graph_image_init.get_poly_graph

        self.world_point_cloud_map = PointCloudMap(self.point_cloud_init,
                                                   types=torch.from_numpy(self.graph_image_init.get_type_indices).to(
                                                       device=self.device),
                                                   remove_node_type=NodeTypes.BORDER,
                                                   fragments_map=fragments_map,
                                                   new_orb_features=self.graph_image_init.attributes_orb.clone().detach(),
                                                   nx_graph=new_graph, device=self.device
                                                   )
        if self.calibrate_cameras:
            self.focal_length_solution, self.cx_solution, self.cy_solution = calibrate_intrinsic_camera_parameters(
                keypoints_3d=self.world_point_cloud_map.points_packed(),
                image_keypoints_2d=self.graph_image_init.get_node_pos_of_crossing_and_end_nodes,
                adjacency=self.world_point_cloud_map.nx_graph.adj_matrix_square,
                r_rotation=self.r_rotation_init, t_translation=self.t_translation_init,
                camera=self.camera, image_size=self.image_size, lr=0.015, iterations=1, device=self.device)

        if self.task['Plot init Graph on 3D Mesh'] or self.task['Make Gif and plot graph on 3D Mesh in .forward() call']:
            self.GifWorldMAp2D = GifMaker(name_of_gif='WorldMapView2D')
            self.GifWorldMApView = GifMaker(name_of_gif='WorldMapView_angles_' + str(40) + '_' + str(200))

            fig_plot_world_3D, ax_plot_world_3D = self.show_scence_from_view(fig=None, ax=None, elev=40, azim=200)
            fig_plot_world_3D.show()

            fig_plot_world_3D, ax_plot_world_3D = self.show_scence_from_view(fig=None, ax=None, elev=0, azim=0)
            fig_plot_world_3D.show()


            save_figure(fig=fig_plot_world_3D,
                        name_of_figure='world_scatter_mesh_in3D_' + str(
                            self.get_nr_of_call) + 'ith_call_angels_40_200_')
            self.GifWorldMApView.add_figure(fig=fig_plot_world_3D)

            fig_plot_world_3D, _ = self.show_scence_from_view(fig=None, ax=None, elev=45, azim=45)
            save_figure(fig=fig_plot_world_3D,
                        name_of_figure='world_scatter_mesh_in3D_' + str(
                            self.get_nr_of_call) + 'ith_call_angels_45_45_', show_fig=True)

            fig_match_graph_with_3D, _ = self.match_graph_with_3D(fig=None, ax=None, graph=self.graph_image_init, matches =self.graph_image_init.get_matches_to_world_map,  elev=30, azim=-45)
            save_figure(fig=fig_match_graph_with_3D,
                        name_of_figure='Fig_Match_Graph_in_3D_with_INIT_Image_Matches_INIT_CALL',
                        show_fig=True, size_width=3.25, size_height=3.25)

    def forward(self, iterations_pose =ITERATION_POSE_OPT, lr_pose=LEARNING_RATE_POSE_OPT, task_update=None, task_texture_update=None):
        if task_update is not None:
            self.task.update(task_update)
        self.set_next_image()  # load next image

        if self.task['Graph based Feature Assignment - for 3d matching and reconstructions']:
            current_image_features = self.current_graph_image.get_orb_attributes()  # features of current image
            self.world_point_cloud_map._remove_nodes_of_type(remove_node_type=NodeTypes.BORDER)
            global_world_feature_tuple = self.world_point_cloud_map.get_orb_feature_matrix
            matches_brute_force = self.match_orb_matrix(global_world_feature_tuple, current_image_features)
            # RANSAC in 3D
            keypoints_3d = self.world_point_cloud_map.points_packed()[matches_brute_force[:, 0]]
            image_keypoints_2d = self.current_graph_image.node_pos[matches_brute_force[:, 1]]
            print('Number of brute force matches: ', matches_brute_force.shape)

            r_rotation_opt, t_translation_opt, inlier_list = triangulate_ransac(keypoints_3d, image_keypoints_2d,
                                                                                r_rotation_init=self.r_rotation_latest.clone(),
                                                                                t_translation_init=self.t_translation_latest.clone(),
                                                                                camera=self.camera,
                                                                                image_size=self.image_size,
                                                                                n_sample_iters=self.task['3D RANSAC number of RANSAC samples that are tested'],
                                                                                ransac_iters_per_samples=self.task['3D RANSAC number of iterations per sample'],
                                                                                reprojection_error_epsilon= self.task['3D RANSAC reporjection error epsilon'],
                                                                                direct_optimization=True,
                                                                                device=self.device)


            unmatched_nodes_in_image = torch.ones(self.current_graph_image.node_pos.shape[0], dtype=torch.bool)
            ransac_match = matches_brute_force[inlier_list]
            print('Ransac matches: ', ransac_match)

            self.world_point_cloud_map.add_new_features_to_existing_points(new_features=current_image_features,
                                                                           matches=ransac_match)

            unmatched_nodes_in_image[matches_brute_force[:, 1]] = False  ## entry in vector is true, if there is no found match, and the corresponding point has to be added a new point to the global map
            r_rotation_node_pose_opt, t_translation_node_pose_opt = self.ModelCameraPoseOpt.camera_pose_forward(matches=ransac_match, meshes=self.mesh_model,
                                                                world_point_cloud_map=self.world_point_cloud_map,
                                                                current_image_node_pos=self.current_graph_image.node_pos,
                                                                r_rotation_init=r_rotation_opt.clone(),
                                                                t_translation_init=t_translation_opt.clone(),
                                                                iterations=iterations_pose, lr=lr_pose)

            # self.task['NodePoseOpt - learning rate']
            # self.task['NodePoseOpt - number of iterations']

            self.r_rotation_latest, self.t_translation_latest = r_rotation_node_pose_opt, t_translation_node_pose_opt
            #self.r_rotation_latest, self.t_translation_latest =  r_rotation_opt, t_translation_opt
            # ToDo: --> following line need to be removed!!
            #self.r_rotation_latest, self.t_translation_latest = self.r_rotation_init, self.t_translation_init

            self.current_graph_image.set_matches_with_world_map(ransac_match)
        else:
            self.r_rotation_latest, self.t_translation_latest = self.r_rotation_init, self.t_translation_init #ToDo: adapt to take always the previous pose reconstruction!

        self.current_graph_image.update_cam_position(self.r_rotation_latest.clone(), self.t_translation_latest.clone())
        self.R_latest, self.T_latest = self.current_graph_image.get_Rotation_and_Translation
        self._add_current_graph_image_to_seen_images(self.current_graph_image)

        if self.task["texture based pose optimization"]:
            self.forward_texture_usage(task_texture_update=task_texture_update) # is using the updated seen_images!


        # for testing!
        # log_R_absolute = r_rotation_opt.clone()
        # log_R_absolute[:, 0]= 0.85* np.pi
        # log_R_absolute[:, 1]= 0.8
        # log_R_absolute[:, 2]= -0.6 #-0.6
        # print('Current blur radius: ', self.inverse_renderer.blur_radius)
        #self.inverse_renderer.blur_radius = 0.0
        # r_param_rotation_opt, t_param_translation_opt = self.ModelCameraPoseOpt.camera_pose_forward(matches=ransac_match, meshes=self.mesh_model,
        #                                                     world_point_cloud_map=self.world_point_cloud_map,
        #                                                     current_image_node_pos=self.current_graph_image.node_pos,
        #                                                     r_rotation_init=log_R_absolute,
        #                                                     t_translation_init=t_translation_opt.clone(),
        #                                                     iterations =ITERATION_POSE_OPT, lr=LEARNING_RATE_POSE_OPT)
        # self.inverse_renderer.blur_radius = 0.04
        # print('Current blur radius: ', self.inverse_renderer.blur_radius)
        #
        # log_R_absolute = r_rotation_opt.clone()
        # log_R_absolute[:, 0]= 0.85* np.pi
        # log_R_absolute[:, 1]= 0.8
        # log_R_absolute[:, 2]= -0.6 #-0.6
        # self.r_rotation_latest = r_param_rotation_opt.clone()
        # self.t_translation_latest = t_param_translation_opt.clone()

        # update map!
        if self.task['Graph based Feature Assignment - for 3d matching and reconstructions']:
            if not torch.all(unmatched_nodes_in_image):
                new_observed_landmarks, fragments_map = self.inverse_renderer(meshes_world=self.mesh_model, node_pos=self.current_graph_image.node_pos[unmatched_nodes_in_image],
                                                                              R=self.R_latest, T=self.T_latest, req_gradient=True, req_fragments=True)  # just for reprojection, no blur_radius is needed!
                # add the new observed landmarks to the world point cloud and aslo update all the newly observed features!
                mask = torch.logical_not(torch.isnan(new_observed_landmarks.points_packed()))[:, 0]
                print('Number of new observed points that are added to the world map: ',
                      current_image_features[unmatched_nodes_in_image][mask].shape)
                new_observed_landmarks = PointCloudMap(new_observed_landmarks, types=torch.from_numpy(self.current_graph_image.get_type_indices).to(device=self.device)[unmatched_nodes_in_image],
                                                       fragments_map=fragments_map, new_orb_features=current_image_features[unmatched_nodes_in_image], device=self.device)
                # Be careful with truncating point clouds!
                # exclusively for new points possible, since the ransac matches are stored to enable an efficient loop closure!
                # --> indices of 3d points must not change after matching
                new_observed_landmarks = new_observed_landmarks.get_sub_point_cloud(mask)
                print('size before extending', self.world_point_cloud_map.points_packed().shape[0])
                self.world_point_cloud_map.add_points(new_observed_landmarks)
                print('size after extending', self.world_point_cloud_map.points_packed().shape[0])

            # ToDo: the following can be  capsulated in a function
            adjacency_matrix = np.uint8(self.current_graph_image.adj.copy())
            new_graph = nx.from_numpy_matrix(adjacency_matrix)

            indices = np.vstack((np.arange(0, unmatched_nodes_in_image.shape[0]), np.arange(0, unmatched_nodes_in_image.shape[0]))).astype(int).transpose()
            indices_of_new_nodes = torch.from_numpy(indices)[unmatched_nodes_in_image][mask]

            size_of_new_points = unmatched_nodes_in_image.shape[0]
            indices_current_graph = np.arange(0, size_of_new_points).astype(int).transpose()
            indices_current_graph = torch.from_numpy(indices_current_graph)
            indices_current_graph = indices_current_graph[unmatched_nodes_in_image][mask]
            indices_global_graph = torch.arange(0, indices_current_graph.shape[0])
            indices_of_new_nodes = torch.vstack((indices_global_graph, indices_current_graph)).t()

            self.world_point_cloud_map.update_graph(new_graph=new_graph, matches_to_world_point_could=ransac_match,
                                                    indices_of_new_nodes=indices_of_new_nodes)

            if self.task['Make Gif and plot graph on 3D Mesh in .forward() call']:
                fig_plot_world_3D, _ = self.show_scence_from_view(fig=None, ax=None, elev=40, azim=200)
                save_figure(fig=fig_plot_world_3D,
                            name_of_figure='world_scatter_mesh_in3D_' + str(
                                self.get_nr_of_call) + 'ith_call_angels_30_200_')
                self.GifWorldMApView.add_figure(fig=fig_plot_world_3D)

                fig_plot_world_3D, _ = self.show_scence_from_view(fig=None, ax=None, elev=30, azim=-45)
                save_figure(fig=fig_plot_world_3D,
                            name_of_figure='world_scatter_mesh_in3D_' + str(
                                self.get_nr_of_call) + 'ith_call_angels_45_45_', show_fig=True)

                fig_world_map_2D = self.plot_world_graph_in2D()
                self.GifWorldMAp2D.add_figure(fig_world_map_2D)
                save_figure(fig=fig_world_map_2D, name_of_figure='world_map_2D_' + str(self.get_nr_of_call) + 'ith_call')

                #fig, ax = generate_mpl_3D_figure(size_width=7.25, size_height=7.25)
                fig_match_graph_with_3D, _ = self.match_graph_with_3D(fig=None, ax=None, graph=self.graph_image_init, matches =self.graph_image_init.get_matches_to_world_map,  elev=30, azim=-45)
                save_figure(fig=fig_match_graph_with_3D,
                            name_of_figure='Fig_Match_Graph_in_3D_with_INIT_Image_Matches_' + str(
                                self.get_nr_of_call) + 'ith_call',
                            show_fig=False)

                #fig, ax = generate_mpl_3D_figure(size_width=7.25, size_height=7.25)
                fig_match_graph_with_3D, _ = self.match_graph_with_3D(fig=None, ax=None, graph=self.current_graph_image, matches =self.current_graph_image.get_matches_to_world_map,  elev=30, azim=-45)
                save_figure(fig=fig_match_graph_with_3D, name_of_figure='Fig_Match_Graph_in_3D_with_current_Image_Matches_' + str(self.get_nr_of_call) + 'ith_call',
                        show_fig=False)

    def forward_texture_usage(self, task_texture_update=None, task_deformation_update=None):

        self.losses_texture_pose_reconstruction = {"chamfer single iteration loss": {"weight": 1.2, "values": []},
                               "chamfer every iteration loss": {"weight": 1.0, "values": []},
                               "image pattern loss": {"weight": 100.0, "values": []},
                               "texture reconstruction loss": {"weight": 12.0, "values": []},
                               "chamfer loss 3d data": {"weight": 10.0, "values": []},}

        self.task_pose_from_texture = {"texture learning": False,
                     "pose by texture": True,
                     #"fix first pose": True,
                     "fix first pose": False,

                     "use chamfer loss with single point cloud extraction": True,
                     "use chamfer loss with point cloud extraction over every iteration": False,
                     "use chamfer loss": None,
                     "use image pattern loss for pose": True,

                     "use chamfer loss - make plots": True,
                     "use chamfer loss - make plots - all variations": True,
                     "use chamfer loss - make plots - all variations in period of": 10,
                     "texture use world map node points for masking chamfer pcs": True,
                     "batch size for rendering": 1,
                     "number of last images 2b used": 1, #2
                     "learning rate - texture": 4*1e-2,
                     "learning rate - pose": 1*1e-3,
                     'Texture based pose opt - number of iterations':34}

        self.task_pose_from_texture.update({"use chamfer loss": self.task_pose_from_texture["use chamfer loss with single point cloud extraction"] or self.task_pose_from_texture["use chamfer loss with point cloud extraction over every iteration"],
                                            })
        if task_texture_update is not None:
            self.task_pose_from_texture.update(task_texture_update)
        # self.renderer_settings = {"face per pixel - opt": 35,
        #                               "blur radius - opt": 1*1e-4,
        #                               "blend param sigma - opt": 1e-5,
        #                               "blend param gamma - opt": 2*1*1e-4,
        #                               "face per pixel - view": 1,
        #                               "blur radius - view": 1e-5,
        #                               "blend param sigma - view": None,
        #                               "blend param gamma - view": None,
        #                               "face per pixel - inverse": 15,
        #                               "blur radius - inverse": 1*np.log(1. / 1e-4 - 1.)*1e-4,
        #                               "blend param sigma - inverse": 1e-5,
        #                               "blend param gamma - inverse": 1*1e-4}

        self.renderer_settings = {
                                # "face per pixel - opt": 1,
                                #   "blur radius - opt": np.log(1. / 1e-4 - 1.)*1e-8,
                                #   "blend param sigma - opt": 1e-8,
                                #   "blend param gamma - opt": 1e-8,
                                  "face per pixel - opt": 25,
                                  "blur radius - opt": 1e-6,
                                  "blend param sigma - opt":  1e-6, #-4
                                  "blend param gamma - opt":  1e-7, #-5

                                  "face per pixel - high opt": 30,
                                  "blur radius - high opt": np.log(1. / 1e-4 - 1.)*1e-4, #-2
                                  "blend param sigma - high opt": 1*1e-4, #-3
                                  "blend param gamma - high opt": 4 * 1e-5, #-1

                                      "face per pixel - view": 1e-6,
                                      "blur radius - view": 1e-5,
                                      "blend param sigma - view": None,
                                      "blend param gamma - view": None,

                                      "face per pixel - inverse": 1, #14 #ToDo check if this is feasible
                                      "blur radius - inverse": 1*np.log(1. / 1e-4 - 1.)*1e-5,
                                      "blend param sigma - inverse": 1e-5,
                                      "blend param gamma - inverse": 1*1e-6}
        # self.renderer_settings = {"face per pixel - opt": 55,
        #                               "blur radius - opt": 1e-4,
        #                               "blend param sigma - opt": 1e-5,
        #                               "blend param gamma - opt": 2*1e-4,
        #                               "face per pixel - view": 1,
        #                               "blur radius - view": 1e-5,
        #                               "blend param sigma - view": None,
        #                               "blend param gamma - view": None,
        #                               "face per pixel - inverse": 15,
        #                               "blur radius - inverse": np.log(1. / 1e-4 - 1.)*1e-4,
        #                               "blend param sigma - inverse": 1e-6,
        #                               "blend param gamma - inverse": 1e-5}

        # self.TextureModel.forward(seen_images=self.seen_images, losses=self.losses_texture_pose_reconstruction, task=self.task_pose_from_texture, renderer_settings=self.renderer_settings, target_images=None,
        #                           world_point_cloud_map=self.world_point_cloud_map, node_point_container_extern=None,
        #                           target_cameras=None, iterations=1,
        #                           lr=0.5 * 1e-3)

        self.TextureModel.forward(seen_images=self.seen_images, losses=self.losses_texture_pose_reconstruction, task=self.task_pose_from_texture, renderer_settings=self.renderer_settings, target_images=None,
                                  world_point_cloud_map=self.world_point_cloud_map, node_point_container_extern=None,
                                  target_cameras=None, iterations=self.task_pose_from_texture['Texture based pose opt - number of iterations'],
                                  lr=0.5 * 1e-3)
        self.TextureModel.plot_losses(as_log=False)

        self.seen_images = self.TextureModel.get_updated_cam_positions_in_seen_images()

        if self.task["deformation - use remaining error for deformation"]:
            task_deformation_update2 = {
                "deformation - make 3D mesh plot after calling remaining error in world_map": False}
            if task_deformation_update is not None:
                task_deformation_update2.update(task_deformation_update)
            self.deform_remaining_texture_error(task_deformation_update=task_deformation_update2)

        # learn texture features
        self.losses_texture_reconstruction = {"chamfer single iteration loss": {"weight": 1.0, "values": []},
                               "chamfer every iteration loss": {"weight": 1.0, "values": []},
                               "image pattern loss": {"weight": 20.0, "values": []},
                               "texture reconstruction loss": {"weight": 20, "values": []}}

        self.task_texture_learning = {"texture learning": True,
                     "pose by texture": False,
                     "fix first pose": True,
                     "use image pattern loss for pose":False,
                     "use chamfer loss with single point cloud extraction": False,
                     "use chamfer loss with point cloud extraction over every iteration": False,
                     "use chamfer loss - make plots": False,
                     "use chamfer loss - make plots - all variations": False,
                     "batch size for rendering": 1,
                     "number of last images 2b used": 2,
                     "learning rate - texture": 2* 1e-1,
                     "learning rate - pose": 2*1e-3,
                                      'Texture Learning number of iterations': 60}

        self.task_texture_learning.update({"use chamfer loss": self.task_texture_learning["use chamfer loss with single point cloud extraction"] or self.task_texture_learning["use chamfer loss with point cloud extraction over every iteration"]})
        if task_texture_update is not None:
            self.task_texture_learning.update(task_texture_update)

        self.renderer_settings = {"face per pixel - opt": 25,# 25 #45
                                  "blur radius - opt": 1e-7,
                                  "blend param sigma - opt": 1e-8,
                                  "blend param gamma - opt": 1*1e-8,
                                  "face per pixel - view": 1,
                                  "blur radius - view": 1e-8,
                                  "blend param sigma - view": None,
                                  "blend param gamma - view": None}
        self.losses_texture = {"texture": {"weight": 1.0, "values": []}}
        self.TextureModel.forward(seen_images=self.seen_images,
                                  losses=self.losses_texture_reconstruction, task=self.task_texture_learning, renderer_settings=self.renderer_settings, target_images=None, target_cameras=None,
                                  iterations=self.task_texture_learning['Texture Learning number of iterations'],
                                  lr=1 * 1e-1)
        self.TextureModel.TextureModel.visualize_texture_prediction()
        self.TextureModel.TextureModel.visualize_mesh()


        # self.mesh_model_textured = self.TextureModel.get_textured_mesh
        # self.seen_images, r_rotation_batch, t_translation_batch = self.CameraPose_RGB.forward(mesh_model_vertex_rgb = self.mesh_model_textured, last_images_2bused=None,
        #                                                     iterations =45, lr=1e-4)
        #
        # R_batch = self.cam_rodrigues_object.get_rotation_matrix(r_rotation_batch)
        # T_batch = self.cam_rodrigues_object.get_translation_matrix(t_translation_batch)
        # cameras = PerspectiveCameras(device=self.device, R=R_batch,
        #                    T=T_batch, focal_length=self.focal_length)
        # images = []
        # for i, graph_images in enumerate(self.seen_images[: self.CameraPose_RGB.get_last_images_applied]):
        #     graph_image = graph_images[2]
        #     _, image_rgb, image_grey =  self.UpComingDataGenerator.load_and_create_graph_image_object(
        #         desired_batch_nr=graph_images[0],
        #         item_in_batch=graph_images[1],)
        #     images.append(image_rgb)
        #
        # target_images = torch.tensor(images,  device=self.device)
        # self.TextureModel.forward(target_images=target_images, target_cameras=cameras, iterations=80, lr=1e-1)

    def deform_remaining_texture_error(self, data_rotation_translation_batch_lists =None,
                                       data_target_imag_batch=None, data_mask_batch_lists=None, deform_loss_new=None, node_point_container_extern=None, task_deformation_update=None):
        deform_loss = {
            "texture chamfer loss": {"weight": 1.0, "values": []},
            "pure texture": {"weight": 0.0, "values": []},
            "edge": {"weight": 0.4, "values": []},
            "normal": {"weight": 0.4, "values": []},
            "laplacian": {"weight": 0.4, "values": []},
        }
        if deform_loss_new is not None:
            deform_loss.update(deform_loss_new)
        if task_deformation_update is not None:
            self.task.update(task_deformation_update)

        # target_image_torch = torch.tensor(img_grey / 255, device=self.device)[None]
        # target_image_mask = self.current_graph_image.generate_mask_for_image(return_as_torch=True)
        # node_pose_mask = self.current_graph_image.generate_mask_for_image(return_as_torch=True)

        renderer_settings_deformation = {
            "face per pixel - opt": 20,
            "blur radius - opt": 1e-7,
            "blend param sigma - opt": 1e-4,
            "blend param gamma - opt": 1e-5,

            "face per pixel - high opt": 40,
            "blur radius - high opt": np.log(1. / 1e-4 - 1.) * 1e-2,
            "blend param sigma - high opt": 1 * 1e-3,
            "blend param gamma - high opt": 4 * 1e-1,

            "face per pixel - view": 1,
            "blur radius - view": 1e-6,
            "blend param sigma - view": None,
            "blend param gamma - view": None,

            "face per pixel - inverse": 1,  # 14 #ToDo check if this is feasible
            "blur radius - inverse": 1 * np.log(1. / 1e-4 - 1.) * 1e-5,
            "blend param sigma - inverse": 1e-5,
            "blend param gamma - inverse": 1 * 1e-5}

        deformed_mesh = self.DeformationModel.forward_deform( geometry_mesh=self.mesh_model,
                                              texture=self.TextureModel.get_textured_mesh.textures,
                                              task=self.task,
                                              seen_images=self.seen_images,
                                              losses= deform_loss,
                                              renderer_settings= renderer_settings_deformation,
                                              data_rotation_translation_batch_lists=data_rotation_translation_batch_lists,
                                              data_target_imag_batch=data_target_imag_batch,
                                              data_mask_batch_lists=data_mask_batch_lists,
                                              target_point_cloud_and_cam_perspective=None,
                                              use_last_k_images_for_deformation=None,
                                              world_point_cloud_map=self.world_point_cloud_map,
                                              node_point_container_extern=node_point_container_extern,
                                              )
        self.update_mesh(new_mesh= deformed_mesh)
        if self.task["deformation - make 3D mesh plot after calling remaining error in world_map"]:
            fig, ax = generate_mpl_3D_figure(size_width=3.5, size_height=3.5, do_tight_fit=False)
            #fig_plot_world_3D, ax = self.show_scence_from_view(fig=fig, ax=ax, elev=40, azim=200)
            fig_plot_world_3D, ax = self.show_scence_from_view(fig=fig, ax=ax, elev=25, azim=35)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1.5])
            #fig_plot_world_3D.tight_layout()
            fig_plot_world_3D.show()
            ax.set_axis_off()
            fig_plot_world_3D.show()
            save_figure(fig=fig_plot_world_3D, name_of_figure='world_scatter_mesh_in3D_after_synthetic_deformation_test_init_call', do_tight_fit = False)


    def init_texture_usage_for_viusalization(self, renderer_settings_texture_visualization_update =None):

        fine_mesh_texture = subdivideMesh(mesh=self.mesh_model.clone(),
                                          iter=self.task["subdivide level for textured mesh - pure visualization"])

        renderer_settings_texture_visualization_tmp = {"face per pixel - opt": 50,
                           "blur radius - opt": 1.5 * 1e-5,
                           "blend param sigma - opt": 1e-5,
                           "blend param gamma - opt": 4 * 1e-6,
                           "face per pixel - view": 1,
                           "blur radius - view": 1e-6,
                           "blend param sigma - view": None,
                           "blend param gamma - view": None,
                           "face per pixel - inverse": 1,
                           "blur radius - inverse": np.log(1. / 1e-4 - 1.) * 1e-5,
                           "blend param sigma - inverse": 1e-5,
                           "blend param gamma - inverse": 1e-4,}


        losses_texture_for_visualization = {"texture": {"texture reconstruction loss": 0.6, "values": []}}
        self.task_texture_for_visualizations = self.task
        self.task_texture_for_visualizations.update({"texture learning": True,
                                                "texture type for reconstruction": TextureTypes.VERTEX_TEXTURE_RGB,
                     "pose by texture": False,
                     "fix first pose": True,
                     "use chamfer loss": False,
                     "use chamfer loss - make plots": True,
                     "use chamfer loss - make plots - all variations": False,
                     "deformation use world map node points for masking chamfer pcs": True,
                     "batch size for rendering": 1,
                     "number of last images 2b used": 3,
                     "learning rate - texture": 1e-3,
                     "learning rate - pose": 2*1e-3})

        renderer_settings_texture_visualization = self.renderer_settings
        renderer_settings_texture_visualization.update(renderer_settings_texture_visualization_tmp)
        if renderer_settings_texture_visualization_update is not None:
            renderer_settings_texture_visualization.update(renderer_settings_texture_visualization_update)

        self.TextureModelVisualization = Texture(mesh=fine_mesh_texture, camera_texture=self.camera_init,  RenderersCollection= self.RenderersCollection,
                                    losses=losses_texture_for_visualization, renderer_settings=renderer_settings_texture_visualization, task=self.task_texture_for_visualizations,
                                    UpComingDataGenerator=self.UpComingDataGenerator, device=self.device)

    def forward_texture_for_viusalization(self, target_images_np: np.ndarray, image_mask_np: np.ndarray,
                                          target_cameras: PerspectiveCameras, task_texture_update=None, renderer_settings_texture_visualization=None):
        if self.TextureModelVisualization is None:
            self.init_texture_usage_for_viusalization(renderer_settings_texture_visualization)

        #losses_texture_for_visualization = {'texture reconstruction loss': {"weight": 0.6, "values": []}}

        self.task_texture_for_visualizations.update({"texture learning": True,
                                                     "number of last images 2b used": 3,
                                                     "learning rate - texture": 1*1e-1,
                                                     "learning rate - pose": 2*1e-3})
        if task_texture_update is not None:
            self.task_texture_for_visualizations.update(task_texture_update)

        # self.renderer_settings = {"face per pixel - opt": 20,
        #                           "blur radius - opt": 1e-5,
        #                           "blend param sigma - opt": 1e-6,
        #                           "blend param gamma - opt": 1e-5,
        #                           "face per pixel - view": 10,
        #                           "blur radius - view": 1e-5,
        #                           "blend param sigma - view": None,
        #                           "blend param gamma - view": None}
        self.losses_texture = {"texture reconstruction loss": {"weight": 0.8, "values": []}}

        # {"chamfer single iteration loss": {"weight": 1.0, "values": []},
        #  "chamfer every iteration loss": {"weight": 1.0, "values": []},
        #  "image pattern loss": {"weight": 1.0, "values": []},
        #  "texture reconstruction loss": {"weight": 1.0, "values": []}}

        if self.task['Texture Reconstruction - load input data from seen_images else function parameters have to be set!']:
            self.TextureModelVisualization.forward(seen_images=self.seen_images, last_images_2bused=8,
                                                   losses=self.losses_texture,
                                                   task=self.task_texture_for_visualizations, renderer_settings=None,
                                                   target_images=None, target_cameras=None,
                                                   iterations=self.task_texture_for_visualizations['Texture Learning number of iterations'],
                                                   lr=1 * 1e-1)

        else:
            self.TextureModelVisualization.forward(target_images=torch.tensor(target_images_np / 255, device=self.device)[None],
                                      image_mask= torch.tensor(image_mask_np, device=self.device)[None].type(dtype=torch.bool),  #self.current_graph_image.generate_mask_for_image(return_as_torch=True)
                                      target_cameras=target_cameras,
                                      losses=self.losses_texture,
                                      task=self.task_texture_for_visualizations, iterations=self.task_texture_for_visualizations['Texture Learning number of iterations'],
                                      lr=5 * 1e-2)

        self.TextureModelVisualization.TextureModel.visualize_texture_prediction()
        self.TextureModelVisualization.TextureModel.visualize_mesh_over_batch(title='Model with texture final solution', batch_size=self.task[
            'Texture Model - batch size of mesh visualization'])


    def set_next_image(self):
        matches, self.desired_batch_nr_ofcurrentinterest,\
        self.desired_item_in_batch_ofcurrentinterest = self.UpComingDataGenerator.next_image_of_interest(
                                                                           current_graph_image=self.current_graph_image,
                                                                           current_image_rgb=self.current_image_rgb,
                                                                           current_image_grey=self.current_image_grey,
                                                                           nr_of_call=self.get_nr_of_call, task_update= self.task)
        new_graph_image, image_rgb_new, image_grey_new = self.UpComingDataGenerator.load_and_create_graph_image_object(
            desired_batch_nr=self.desired_batch_nr_ofcurrentinterest,
            item_in_batch=self.desired_item_in_batch_ofcurrentinterest,
            r_rotation=None,
            t_translation=None)
        # fig, ax = generate_mpl_figure()
        # ax.imshow(image_rgb_new)
        # fig.show()

        if self.task['Plot matches between current and newly loaded graph image - world_map']:
            fig_matches_with_next_image = plot_graph_matches_color(self.current_image_rgb, image_rgb_new,
                                                                   copy.deepcopy(self.current_graph_image.node_pos),
                                                                   self.current_graph_image.adj,
                                                                   copy.deepcopy(new_graph_image.node_pos),
                                                                   new_graph_image.adj,
                                                                   matches, color_edge=None)
            save_figure(fig=fig_matches_with_next_image, name_of_figure='matches_with_next_image' +
                                                                        str(self.get_nr_of_call) + '_ith_call_')
            print('Currently loaded item in batch', self.desired_item_in_batch_ofcurrentinterest)
            print('Currently loaded batch nr', self.desired_batch_nr_ofcurrentinterest)
            print('Number of matches: ', matches.shape[0])
        self.current_graph_image = new_graph_image
        self.current_image_rgb = image_rgb_new
        self.current_image_grey = image_grey_new
        self.seen_images.insert(0, [self.desired_batch_nr_ofcurrentinterest, self.desired_item_in_batch_ofcurrentinterest,
                                 True])
        self.seen_images_of_intrest.insert(0,
            [self.desired_batch_nr_ofcurrentinterest, self.desired_item_in_batch_ofcurrentinterest])

        # self.seen_images.append([self.desired_batch_nr_ofcurrentinterest, self.desired_item_in_batch_ofcurrentinterest,
        #                          True])
        # self.seen_images_of_intrest.append(
        #     [self.desired_batch_nr_ofcurrentinterest, self.desired_item_in_batch_ofcurrentinterest])

    def update_point_cloud(self):
        self.point_cloud_world_init.update_padded(new_points_padded=None, new_normals_padded=None,
                                                  new_features_paddeed=None)  # ToDo implement pontcloud update with merging the current pointcloud to the world point cloud

    def get_features(self, current_gim):
        features = torch.full((current_gim.node_pos.shape[0], 3), 0.5, device=self.device)
        return current_gim.get_features  # ToDo give gim class a feature container

    def match_orb(self, global_world_map_features, current_image_features):
        # match = [np.arange(global_world_features.shape[0]),np.arange(current_features.shape[0])]
        matches_brute_force = match(global_world_map_features, current_image_features, cross_check=True,
                                    max_distance=np.inf)
        unmatched_landmarks = None
        return matches_brute_force

    def match_orb_matrix(self, global_world_features, current_image_features):
        global_world_feature_matrix = global_world_features[0]
        global_world_feature_matrix_hash = global_world_features[1]
        lastest_global_features = global_world_feature_matrix[global_world_feature_matrix_hash[:, 0], :,
                                  global_world_feature_matrix_hash[:, 1] - 1].clone().detach().cpu()

        matches_brute_force = match(lastest_global_features, current_image_features, cross_check=True,
                                    max_distance=np.inf)
        unmatched_landmarks = None
        return matches_brute_force

    def _add_current_graph_image_to_seen_images(self, current_graph_image):
        assert self.seen_images[0][2]
        self.seen_images[0][2] = current_graph_image
        # assert self.seen_images[-1][2]
        # self.seen_images[-1][2] = current_graph_image



    def create_world_map_camera_scene_figure(self, fig=None, ax=None, size_width=3.5, size_height=3.5):

        if fig is None and ax is None:
            fig, ax = generate_mpl_3D_figure(size_width=size_width, size_height=size_height)
        fig, ax = create_world_map_mesh_and_graph_figure(world_point_cloud_map=self.world_point_cloud_map, mesh_model=self.mesh_model, fig=fig, ax=ax)

        cameras = self.get_seen_cameras()
        # ToDo reinit would solved the issue here!
        # c = PerspectiveCameras(R=cameras.R[:], T=cameras.T[:])
        handle_cam = plot_cameras(ax, cameras, color=blue_5)

        fig, ax = plot_mesh(self.mesh_model, fig=fig, ax=ax, alpha_nodes=0.3, alpha_edges=0.3)

        return (fig, ax)

    def show_scence_from_view(self, fig=None, ax=None, elev=0, azim=0):
        #ToDo:check why reusing is not working
        if fig is None and ax is None:
            fig, ax = generate_mpl_3D_figure()

        fig, ax = self.create_world_map_camera_scene_figure(fig=fig, ax=ax)

        # Set the initial view
        ax.view_init(elev, azim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # Hide the axes
        # ax.set_axis_off()
        #fig.show()
        return fig, ax


    def match_graph_with_3D(self, graph,  matches, fig=None, ax=None, elev=30, azim=-45):

        if fig is None and ax is None:
            fig, ax = generate_mpl_3D_figure()

        fig, ax = plot_mesh(self.mesh_model, fig=fig, ax=ax, alpha_nodes =0.8, thickness_nodes=0.25*SCALE_FIGURE_SETTINGs, color_edges=COLOR_MESH, alpha_edges =0.65)

        fig, ax = create_world_map_mesh_and_graph_figure(world_point_cloud_map=self.world_point_cloud_map,
                                                          mesh_model=self.mesh_model, fig=fig, ax=ax)

        # Get node positions
        node_positions_on_mesh = self.world_point_cloud_map.points_packed().clone().cpu().detach().numpy()

        R1 = self.cam_rodrigues_object.get_rotation_matrix(
            torch.tensor([0, 0, 0]).to(self.device)).clone().detach().cpu().numpy()
        R2 = self.cam_rodrigues_object.get_rotation_matrix(
            torch.tensor([0, 0, 0]).to(self.device)).clone().detach().cpu().numpy()
        R = R1 @ R2
        # R = self.cam_rodrigues_object.get_rotation_matrix(
        #     torch.tensor([0, 0, 0]).to(self.device)).clone().detach().cpu().numpy()
        # R = self.cam_rodrigues_object.get_rotation_matrix(
        #     torch.tensor([0, 0, 0]).to(self.device)).clone().detach().cpu().numpy()
        scale = 0.01
        image_size = 128
        T_translate3d = np.array([-(image_size / 2) * scale - 0.2, -(image_size / 2) * scale + 0.2, 2.0])

        ## rectangle
        x = [0, image_size, image_size, 0]
        y = [0, 0, image_size, image_size]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]
        v = np.array(verts) * scale
        verts_transformed = np.matmul(v, R) + T_translate3d

        rectangle = Poly3DCollection(verts_transformed)
        rectangle.set_edgecolor(black)
        rectangle.set_facecolor(gray_1)
        rectangle.set_alpha(0.2)
        ax.add_collection3d(rectangle)

        # https://matplotlib.org/3.5.0/gallery/mplot3d/text3d.html
        ax.text(verts_transformed[0][0][0]+0.2, verts_transformed[0][0][1]+0.07, verts_transformed[0][0][2], s="2D", color=black, zdir=(0, 1, 0), fontsize=6)

        # Get node positions
        # node_pos_3d = mesh_model.verts_packed().clone().cpu().detach().numpy()
        positions = graph.node_pos
        Graph = graph.nx_graph

        pos_list = []
        for i in range(len(positions)):
            pos_list.append(
                [positions[i][0],
                 image_size - positions[i][1], 0])  # flip y-axis, since node_pos are in image coordinates
        node_pos_array = np.array(pos_list)

        # node_pos_3d  = np.array(node_pos_array)*scale
        node_pos_3d_image = np.matmul(node_pos_array * scale, R)[0] + T_translate3d

        # Get number of nodes
        n = graph.nx_graph.number_of_nodes()

        # Get the maximum number of edges adjacent to a single node
        edge_max = max([graph.nx_graph.degree(i) for i in range(n)])

        # Define color range proportional to number of edges adjacent to a single node
        # colors = [plt.cm.plasma(graph.nx_graph.degree(i) / edge_max) for i in range(n)]
        node_colors = list()
        # Define color of nodes depending on the node type
        for type in graph.get_node_types:
            if type == NodeTypes.CROSSING.value:
                node_colors.append(NodeTypes.CROSSING.colour)
            elif type == NodeTypes.END.value:
                node_colors.append(NodeTypes.END.colour)
            elif type == NodeTypes.BORDER.value:
                node_colors.append(NodeTypes.BORDER.colour)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in enumerate(node_pos_3d_image):
            xi = value[0]
            yi = value[1]
            zi = value[2]
            # Scatter plot
            ax.scatter(xi, yi, zi, color=node_colors[key], s=3*SCALE_FIGURE_SETTINGs + 2*SCALE_FIGURE_SETTINGs * graph.nx_graph.degree(key),
                       edgecolors=darkslategrey, alpha=0.8, linewidths=0.6*SCALE_FIGURE_SETTINGs)
            #ax.scatter(xi, yi, zi, color=node_colors[key], s=3*SCALE_FIGURE_SETTINGs + 2*SCALE_FIGURE_SETTINGs * graph.nx_graph.degree(key), alpha=1)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(graph.nx_graph.edges()):
            x = np.array((node_pos_3d_image[j[0]][0], node_pos_3d_image[j[1]][0]))
            y = np.array((node_pos_3d_image[j[0]][1], node_pos_3d_image[j[1]][1]))
            z = np.array((node_pos_3d_image[j[0]][2], node_pos_3d_image[j[1]][2]))
            # Plot the connecting lines
            ax.plot(x, y, z, color=black, alpha=0.9, linewidth=1*SCALE_FIGURE_SETTINGs)

        # plot matches from image to mesh location
        # if self.graph_image_init.get_crossing_and_end_nodes_indices is not None:
        #     node_pos_3d_image_corresponding_node_types = node_pos_3d_image[self.graph_image_init.get_crossing_and_end_nodes_indices]

        node_pos_3d_image_corresponding_node_types = node_pos_3d_image
        for i, j in enumerate(matches):
            x = np.array((node_positions_on_mesh[j[0]][0], node_pos_3d_image_corresponding_node_types[j[1]][0]))
            y = np.array((node_positions_on_mesh[j[0]][1], node_pos_3d_image_corresponding_node_types[j[1]][1]))
            z = np.array((node_positions_on_mesh[j[0]][2], node_pos_3d_image_corresponding_node_types[j[1]][2]))
            # Plot the connecting lines
            ax.plot(x, y, z, color=red_3, alpha=1, linewidth=0.8*SCALE_FIGURE_SETTINGs)


        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Set the initial view
        ax.view_init(elev, azim)
        ax.view_init(elev=30, azim=-45)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1.5])
        fig.tight_layout()
        fig.show()

        return fig, ax


    def get_seen_cameras(self):
        with torch.no_grad():
            r_rotation_batch = torch.zeros((self.seen_images.__len__(), 3), dtype=torch.float32, device=self.device)
            t_translation_batch = torch.zeros((self.seen_images.__len__(), 3), dtype=torch.float32, device=self.device)
            for i, graph_images in enumerate(self.seen_images):
                graph_image = graph_images[2]
                r_rotation_batch[i, :] = graph_image.r_rotation
                t_translation_batch[i, :] = graph_image.t_translation
                # if i == 0:
                #     # furthermore, we know that the first camera is a trivial one
                #     self.r_rotation_init = graph_image.r_rotation.clone().detach()
                #     self.t_translation_init = graph_image.t_translation.clone().detach()
            R_batch = self.cam_rodrigues_object.get_rotation_matrix(r_rotation_batch)
            T_batch = self.cam_rodrigues_object.get_translation_matrix(t_translation_batch)
            focal_length = self.inverse_renderer.rasterizer.cameras.focal_length.clone()
            assert T_batch.ndim == 2
            cameras = PerspectiveCameras(focal_length=focal_length, R=R_batch.detach(), T=T_batch.detach(), device=self.device)
            # ToDo delete the following!
            # plot_camera_scene(cameras,cameras_gt=cameras,status='camera poses')
        return cameras

    def get_init_camera_perspective(self, from_graph_image = False):
        with torch.no_grad():
            if not from_graph_image:
                return self.camera_init
            else:
                graph_image = self.seen_images[-1][2]
                R_batch = self.cam_rodrigues_object.get_rotation_matrix(graph_image.r_rotation)
                T_batch = self.cam_rodrigues_object.get_translation_matrix(graph_image.t_translation)
                focal_length = self.inverse_renderer.rasterizer.cameras.focal_length.clone()
                assert T_batch.ndim == 2
                cameras = PerspectiveCameras(focal_length=focal_length, R=R_batch.detach(), T=T_batch.detach(), device=self.device)
                return cameras


    def plot_world_graph_in2D(self, angle=0, save=False):

        fig , ax = generate_mpl_figure()
        projected_image_keypoints_2d = camera_projection(keypoints_3d=self.world_point_cloud_map.points_packed(),
                                                         camera=self.camera_view_only,
                                                         image_size=self.image_size, device=self.device,
                                                         r_rotation=self.r_rotation_init,
                                                         t_translation=self.t_translation_init)
        img_cv = projected_image_keypoints_2d.clone().cpu().detach().numpy()
        # plt.plot(img_cv[:, 0], 128 - img_cv[:, 1], 'ro')  # plot x and y using blue circle markers
        # plt.title('projected keypoints in 2D')
        # plt.show()
        ax.plot(img_cv[:, 0], 128 - img_cv[:, 1], 'ro')  # plot x and y using blue circle markers
        ax.set_title('projected keypoints in 2D')
        fig.tight_layout()
        fig.show()



        # fig = plot_graph(image_size=self.image_size.cpu().detach().numpy(),
        #                  pos = img_cv, nx_graph = self.world_point_cloud_map.nx_graph)
        image_size = self.image_size.clone().cpu().detach().numpy()
        # Get number of nodes
        edges = list(self.world_point_cloud_map.nx_graph.edges(data="ith_observations"))

        # Get the maximum number of edges adjacent to a single node
        seen_max = 4

        # Define edge color by attributes of the edges
        edge_colors = [plt.cm.plasma(edges[i][2] / seen_max) for i in range(edges.__len__())]

        node_colors = list()
        # Define color of nodes depending on the node type
        for type in self.world_point_cloud_map.get_node_types:
            if type == NodeTypes.CROSSING.value:
                node_colors.append(NodeTypes.CROSSING.colour)
            elif type == NodeTypes.END.value:
                node_colors.append(NodeTypes.END.colour)
            elif type == NodeTypes.BORDER.value:
                node_colors.append(NodeTypes.BORDER.colour)

        # 3D network plot
        fig , ax = generate_mpl_figure(size_width=3.25, size_height=3.25)
        positions = img_cv
        Graph = self.world_point_cloud_map.nx_graph

        pos_list = []
        for i in range(len(positions)):
            pos_list.append(
                [positions[i][0],
                 image_size[0] - positions[i][1]])  # flip y-axis, since node_pos are in image coordinates
        p = dict(enumerate(pos_list, 0))
        nx.set_node_attributes(Graph, p, 'pos')
        y_lim = image_size[0]
        x_lim = image_size[1]
        extent = 0, x_lim, 0, y_lim
        # nx.draw(Graph, pos=p, node_size=50, edge_color= edge_colors , width=3, node_color='r')
        nx.draw(Graph, pos=p, node_size=18*SCALE_FIGURE_SETTINGs, edge_color=edge_colors, width=1.5*SCALE_FIGURE_SETTINGs, node_color=node_colors)
        fig.tight_layout()
        fig.show()
        return fig

    def update_mesh(self, new_mesh):
        assert self.mesh_model.num_verts_per_mesh() == new_mesh.num_verts_per_mesh()
        _ = self.world_point_cloud_map.update_point_cloud_based_on_mesh(mesh=new_mesh)
        self.mesh_model = new_mesh

    @property
    def get_current_camera_position(self):
        R_current, T_current = self.current_graph_image.get_Rotation_and_Translation
        self.camera.R = R_current
        self.camera.T = T_current
        return self.camera

    @property
    def get_world_mesh(self):
        return self.mesh_model

    @property
    def get_nr_of_call(self):
        return self.seen_images.__len__()

    @property
    def get_seen_images(self):
        return self.seen_images

    def close_gifs(self):
        if PLOT_WORLD_MAP:
            self.GifWorldMApView.close_writer()
            self.GifWorldMAp2D.close_writer()
        else:
            print('If GIFS are supposed to be created, the variable PLOT_WORLD_MAP in config_plots.py has to be set!')




