import torch
import torch.nn as nn
from configs.plot.colours import *

# io utils

# datastructures

# 3D transformations functions
# rendering components
from pytorch3d.renderer import (
    look_at_rotation
)
from models.pose_rodrigues_rot_fromula import Camera3DPoseRodriguesRotFormula
from utils.camera_visualization import plot_cameras
from utils.world_map_visualization import create_world_map_mesh_and_graph_figure
from configs.plot.config_plots import *

class PoseModel(nn.Module):
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
    def __init__(self, meshes, inverse_renderer, match, current_image_node_pos, world_point_cloud_map,
                 r_param_rotation: torch.tensor, t_param_translation: torch.tensor,
                 losses, ith_call:int=0):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.match = match
        self.inverse_renderer = inverse_renderer
        self.current_node_pos = current_image_node_pos
        self.world_point_cloud = world_point_cloud_map
        self.current_point_cloud = None
        self.mask = None
        self.matched_world = None
        self.matched_current = None
        self.cameras = self.inverse_renderer.rasterizer.cameras.clone()
        if MAKE_GIF_CAMERA_POSE_ERROR:
            self.GifMaker = GifMaker(name_of_gif='camera_pose_opt_'+str(ith_call)+'_ith_call')

        if losses:
            self.losses = losses
        else:
            self.losses = {"euclidean": {"weight": 1.0, "values": []},
                           "normal": {"weight": 0.8, "values": []},
                           "pull-back": {"weight": 0.1, "values": []}
                           }
        self.__define_parameters(r_param_rotation=r_param_rotation,t_param_translation=t_param_translation)
        self.intersection_point = None


    def __define_parameters(self,r_param_rotation: torch.tensor, t_param_translation: torch.tensor):
        # Create an optimizable parameter for the x, y, z position of the camera.
        # Model Parameter:
        self.r_param_rotation = nn.Parameter(r_param_rotation.to(self.device).type(torch.float64))
        self.t_param_translation = nn.Parameter(t_param_translation.to(self.device).type(torch.float64))
        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=self.device)

    @property
    def get_parameters_rotation_and_translation(self):
        return self.r_param_rotation, self.t_param_translation



    def forward(self, iteration=0):
        # Render the image using the updated camera position. Based on the new position of the
        # camera we calculate the rotation and translation matrices
        loss = {k: torch.tensor(0.0, device=self.device) for k in self.losses}

        R = self.cam_rodrigues_object.get_rotation_matrix(self.r_param_rotation)
        T = self.cam_rodrigues_object.get_translation_matrix(self.t_param_translation)

        self.current_point_cloud, _ = self.inverse_renderer(meshes_world=self.meshes.clone(), node_pos=self.current_node_pos, R=R, T=T, req_gradient=True, req_fragments=False)

        self.cameras.R = R
        self.cameras.T = T

        # Calculate the point cloud loss
        mask = torch.logical_not(torch.isnan(self.current_point_cloud.points_packed()[self.match[:, 1]]))[:, 0].cpu().detach()

        self.matched_world = self.match[:, 0][mask]
        self.matched_current = self.match[:, 1][mask]
        if self.matched_current.shape[0] == 0:
            print('No valid matches in view, turn camera back to object')

        loss["euclidean"] = self.euclidean_loss()
        loss["normal"] = self.normal_point_loss()
        if 'pull-back' in loss:
            camera_distance, intersection_distance, intersection_camera_distance = self.pull_out_mesh(
                camera=self.cameras)
            error_pull_back = torch.sigmoid(-40 * (camera_distance - intersection_distance)) * intersection_camera_distance
            loss["pull-back"] = error_pull_back


        # Weighted sum of the losses
        sum_loss = torch.tensor(0.0, device=self.device)
        for k, l in loss.items():
            weighted_loss = l * self.losses[k]["weight"]
            sum_loss += weighted_loss
            self.losses[k]["values"].append(float(weighted_loss.detach().cpu()))

        if torch.isnan(sum_loss) and 'pull-back' in loss:
            sum_loss = error_pull_back

        return sum_loss
               #self.r_param_rotation, self.t_param_translation #ToDo implement an if statement to realize two return function whether we are in the optimization loop or evaluation loop

    def pull_out_mesh(self,camera):
        #ToDo:plot camera and intersection point!
        #ToDo test without rotation!
        cam_center = camera.get_camera_center(eps=0.0001).clone()
        Tt = camera.T.clone()
        R_point_to_origin = look_at_rotation(camera_position = cam_center, at=((0, 0, 0),), up=((0, 1, 0), ), device= self.device)
        image_centre = np.array([[(self.inverse_renderer.image_size / 2), (self.inverse_renderer.image_size / 2)]]).astype(int)
        #image_centre = np.array([[(0), (0)]]).astype(int)
        #ToDo check correct T matrix, by plott!!

        T = torch.matmul(cam_center, R_point_to_origin.inverse())[0]

        camera.R = R_point_to_origin.inverse().clone()
        #camera.T = Tt
## ToDo just plot the sanity check in the last iteration!!
        #self.plot_stitch_in_3D(camera, angle=120, save=False)

        intersection_point, _ = self.inverse_renderer(meshes_world=self.meshes.clone(),
                                                      node_pos=image_centre, R=R_point_to_origin, T=Tt, req_gradient=True, req_fragments=False)

        camera_center = camera.get_camera_center(eps=0.001).clone()
        intersection_distance = (intersection_point.points_packed() ** 2).mean().sqrt()
        camera_distance = (camera_center** 2).mean().sqrt()
        intersection_camera_distance = ((intersection_point.points_packed() - camera_center) ** 2).mean().sqrt()

        # print(intersection_distance.sqrt())
        # print(camera_distance.sqrt())
        # print(intersection_camera_distance.sqrt())
        return camera_distance, intersection_distance, intersection_camera_distance

    def euclidean_loss(self):
        euclidean_loss = ((self.world_point_cloud.points_packed()[self.matched_world]-self.current_point_cloud.points_packed()[self.matched_current])**2) # ToDo increase the error function here if entries are nan!!!
        mask = torch.logical_not(torch.isnan(euclidean_loss))
        return euclidean_loss[mask].mean()

    def normal_point_loss(self):
        normal_loss = (1 - torch.cosine_similarity(self.world_point_cloud.normals_packed()[self.matched_world],
                                                  self.current_point_cloud.normals_packed()[self.matched_current], dim=1))
        mask = torch.logical_not(torch.isnan(normal_loss))
        return normal_loss[mask].mean()

    @property
    def get_loss_dict_history(self):
        return self.losses

    def close_gif_maker(self):
        if MAKE_GIF_CAMERA_POSE_ERROR:
            self.GifMaker.close_writer()

    def plot_pose_error_on_mesh_scene(self, ith_call:int=0, ith_iteration:int=0, fig=None, ax=None):

        if fig is None or ax is None:
            fig, ax = generate_mpl_3D_figure()

        fig, ax = create_world_map_mesh_and_graph_figure(world_point_cloud_map=self.world_point_cloud,
                                                         mesh_model=self.meshes, fig=fig, ax=ax)

        if self.current_point_cloud is None:
            raise Exception('First call forward() function to execute the inverse rendering,'
                            ' before loss can ba visualized!')
        # Get node positions
        world_map_node_points = self.world_point_cloud.points_packed().clone().cpu().detach().numpy()
        reprojected_node_pos = self.current_point_cloud.points_packed().clone().cpu().detach().numpy()


        #   # Define color range proportional to number of edges adjacent to a single node
        # colors = [plt.cm.plasma(world_point_cloud_map.nx_graph.degree(i) / edge_max) for i in range(n)]

        # 3D network plot
        #with plt.style.context(('ggplot')):

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in enumerate(reprojected_node_pos):
            xi = value[0]
            yi = value[1]
            zi = value[2]
            # Scatter plot
            ax.scatter(xi, yi, zi, color=orange_2, s=6*SCALE_FIGURE_SETTINGs, alpha=0.9, linewidths=0.6*SCALE_FIGURE_SETTINGs, edgecolors=black)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted+
        for i in range(self.matched_world.shape[0]):
            x = np.array((world_map_node_points[self.matched_world[i]][0], reprojected_node_pos[self.matched_current[i]][0]))
            y = np.array((world_map_node_points[self.matched_world[i]][1], reprojected_node_pos[self.matched_current[i]][1]))
            z = np.array((world_map_node_points[self.matched_world[i]][2], reprojected_node_pos[self.matched_current[i]][2]))
            # Plot the connecting lines
            ax.plot(x, y, z, color=red_2, linewidth=1*SCALE_FIGURE_SETTINGs, alpha=1.0)

        fig.show()
        if MAKE_GIF_CAMERA_POSE_ERROR:
            self.GifMaker.add_figure(fig=fig)
        if SAVE_LOSS_CAMERA_POSE:
            save_figure(fig=fig, name_of_figure='pose_opt_error'+str(ith_call)+'_ith_call'+str(ith_iteration)+'_ith_itr')

        return fig, ax





    def plot_stitch_in_3D(self, camera, angle=0, save=False, fig=None, ax=None):

        if fig is None or ax is None:
            fig, ax = generate_mpl_3D_figure()
        # Get node positions
        angle = 120
        pos = self.world_point_cloud.points_packed().clone().cpu().detach().numpy()

        # Get number of nodes
        n = self.world_point_cloud.nx_graph.number_of_nodes()

        # Get the maximum number of edges adjacent to a single node
        edge_max = max([self.world_point_cloud.nx_graph.degree(i) for i in range(n)])

        # Define color range proportional to number of edges adjacent to a single node
        colors = [plt.cm.plasma(self.world_point_cloud.nx_graph.degree(i) / edge_max) for i in range(n)]

        # 3D network plot
        # with plt.style.context(('ggplot')):
        #
        #     fig = plt.figure(figsize=(10, 7))
        #     ax = Axes3D(fig)

        vertex_pos = self.meshes.verts_packed().clone().cpu().detach().numpy()
        ax.scatter(vertex_pos[:, 0], vertex_pos[:, 1], vertex_pos[:, 2], color="green",  alpha=0.8, s=1)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in enumerate(pos):
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, c=colors[key], s=10 + 10 * self.world_point_cloud.nx_graph.degree(key), edgecolors='k', alpha=0.9)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(self.world_point_cloud.nx_graph.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, color='black', alpha=0.7)
        # mesh edges
        mesh_edges = self.meshes.edges_packed().clone().cpu().detach().numpy()
        for mesh_edge in mesh_edges:
            x = np.array((vertex_pos[mesh_edge[0]][0], vertex_pos[mesh_edge[1]][0]))
            y = np.array((vertex_pos[mesh_edge[0]][1], vertex_pos[mesh_edge[1]][1]))
            z = np.array((vertex_pos[mesh_edge[0]][2], vertex_pos[mesh_edge[1]][2]))
            # Plot the connecting lines
            ax.plot(x, y, z, color='green', alpha=0.2)
        handle_cam = plot_cameras(ax, camera, color='blue')

        # Plot the pull back line through origin and camera
        origin = np.array([0, 0, 0])
        camera_center = camera.get_camera_center().clone().cpu().detach().numpy()
        x = np.array((camera_center[0][0], origin[0]))
        y = np.array((camera_center[0][1], origin[1]))
        z = np.array((camera_center[0][2], origin[2]))

        ax.plot(x, y, z, color='red', alpha=0.7)
        ax.scatter(origin[0], origin[1], origin[2], color="red", alpha=0.8, s=15)
        ax.scatter(camera_center[0][0], camera_center[0][1], camera_center[0][2], color="red", alpha=0.8, s=15)

        # Set the initial view
        ax.view_init(30, angle)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # Hide the axes
        #ax.set_axis_off()
        if save is not False:
            print('Add save function!!!')
            # plt.savefig("C:\scratch\\data\"+str(angle).zfill(3)+".png")
            # plt.close('all')
        else:
            plt.show()
