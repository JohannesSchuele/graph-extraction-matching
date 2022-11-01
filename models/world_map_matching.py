import numpy as np

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
from tools_generate.UpComingDataGeneratorMatching import UpComingDataGenerator
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

class WorldMapMatching():

    def __init__(self, data_generator):

        self.seen_images = list()
        self.seen_images_of_intrest = list()

        self.data_generator = data_generator
        self.init_graph_image = None
        self.current_graph_image = None
        self.current_image_rgb = None
        self.current_image_grey = None
        self.current_point_cloud = None
        self.R_latest = None
        self.T_latest = None
        self.print_loss = None

        self.current_batch_object = None
        self.currently_loaded_batch_nr = None  # do not touch that variable!! It tracks the currently loaded batch in order to avoid reloading
        self.currently_loaded_batch_item = None  # do not touch that variable!! It tracks the currently loaded batch in order to avoid reloading
        self.desired_item_in_batch_ofcurrentinterest = 0
        self.desired_batch_nr_ofcurrentinterest = 0
        self.world_map_matches = list()
        self.remove_node_type = NodeTypes.BORDER


        self.UpComingDataGenerator = UpComingDataGenerator(data_generator=data_generator, device=device)

        #self.init()
        ## Model Parameter:
        self.TextureModelVisualization = None
        self.renderer_settings = self.RenderersCollection.renderer_settings

        self.init_load_graph_img()

        self.init_distortion()


    def init_distortion(self):

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

        fig, ax = visualize_image(rotated_image, size_width=3.75, size_height=3.75, title='')
        fig.show()

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

        #undist_pos = self.current_graph_image.get_all_image_nodes
        #data_target_imag_batch = torch.from_numpy(self.current_image_grey)[None].to(device=self.device)

        undist_pos, mask_for_image_distorted = distort_image(cameraparams=get_camera_params(),
                                             image=self.current_graph_image.generate_mask_for_image(),
                                             input_pos=self.current_graph_image.get_all_image_nodes, plot_it=True)

        undist_pos, matching_mask_for_image_distorted = distort_image(cameraparams=get_camera_params(),
                                             image=self.current_graph_image.generate_matching_mask_for_image(),
                                             input_pos=self.current_graph_image.get_all_image_nodes, plot_it=True)



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


    def forward(self):

        self.set_next_image()  # load next image
        self._add_current_graph_image_to_seen_images(self.current_graph_image) # could also be moved, however it is currently in the form of the diff rendering pipeline

        # global_world_feature_tuple = self.world_point_cloud_map.get_orb_feature_matrix
        # matches_brute_force = self.match_orb_matrix(global_world_feature_tuple, current_image_features)

        # keypoints_3d = self.world_point_cloud_map.points_packed()[matches_brute_force[:, 0]]
        # image_keypoints_2d = self.current_graph_image.node_pos[matches_brute_force[:, 1]]

        #     adjacency_matrix = np.uint8(self.current_graph_image.adj.copy())
        #     new_graph = nx.from_numpy_matrix(adjacency_matrix)
        #     self.world_point_cloud_map.update_graph(new_graph=new_graph, matches_to_world_point_could=ransac_match,
        #                                             indices_of_new_nodes=indices_of_new_nodes)



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




