import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation)
from tools_graph.utilz_analysis import plot_graph
from models.pose_rodrigues_rot_fromula import Camera3DPoseRodriguesRotFormula
from mpl_toolkits.mplot3d import Axes3D
from utils.camera_visualization import plot_cameras, plot_camera_scene
import cv2




















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
    def __init__(self, meshes,inverse_renderer, match, current_node_pos, world_point_cloud_map, r_param_rotation: torch.tensor, t_param_translation: torch.tensor, weight_normal_loss = 0.1, print_loss = False):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.match = match
        self.inverse_renderer = inverse_renderer
        self.current_node_pos = current_node_pos
        self.world_point_cloud = world_point_cloud_map
        self.current_point_cloud = None
        self.mask = None
        self.matched_world = None
        self.matched_current = None
        # Create an optimizable parameter for the x, y, z position of the camera.
        # Model Parameter:
        self.r_param_rotation = nn.Parameter(r_param_rotation.to(self.device))
        self.t_param_translation = nn.Parameter(t_param_translation.to(self.device))
        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=self.device)

        self.weight_normal_loss = weight_normal_loss
        self.print_loss = print_loss

    def forward(self):
        # Render the image using the updated camera position. Based on the new position of the
        # camera we calculate the rotation and translation matrices
        #R, T = look_at_view_transform(dist=self.dist, elev=self.elev, azim=self.azim, device=self.device)
        R = self.cam_rodrigues_object.get_rotation_matrix(self.r_param_rotation)
        T = self.cam_rodrigues_object.get_translation_matrix(self.t_param_translation)
        self.current_point_cloud, _ = self.inverse_renderer(meshes_world=self.meshes.clone(), node_pos=self.current_node_pos, R=R, T=T, req_gradient=True, req_fragments=False)
        self.inverse_renderer.rasterizer.cameras.R = R
        self.inverse_renderer.rasterizer.cameras.T = T

        # Calculate the point cloud loss
        #self.current_point_cloud.points_packed()
        mask = torch.logical_not(torch.isnan(self.current_point_cloud.points_packed()[self.match[:, 1]]))[:, 0].cpu().detach()
        #mask = copy.deepcopy(mask).cpu().detach().numpy() #ToDo check if deepcopy is necessary
        # #self.mask = mask[:, 0] # Since we are interested in the general existence
        # self.matched_world = self.match[:, 0][mask].cpu()
        # self.matched_current = self.match[:, 1][mask].cpu()
        #mask = copy.deepcopy(mask[:,0]).cpu().detach().numpy() #ToDo check if deepcopy is necessary
        #self.mask = mask[:, 0] # Since we are interested in the general existence
        #self.matched_world = self.match.cpu().detach().numpy()[:, 0][mask]
        #self.matched_current = self.match.cpu().detach().numpy()[:, 1][mask]
        self.matched_world = self.match[:, 0][mask]
        self.matched_current = self.match[:, 1][mask]
        if self.matched_current.shape[0] == 0: ##ToDo this acutall needs to be smaller than 6
            print('No valid matches in view, turn camera back to object')
        return self.loss(camera=self.inverse_renderer.rasterizer.cameras),\
               self.r_param_rotation, self.t_param_translation #ToDo implement an if statement to realize two return function whether we are in the optimization loop or evaluation loop

    def loss(self,camera):
        euclidean_loss = self.euclidean_loss()
        normal_loss = self.normal_point_loss()

        camera_distance, intersection_distance, intersection_camera_distance = self.pull_out_mesh(camera=camera)
        error_pull_back = torch.sigmoid(-40*(camera_distance-intersection_distance))*intersection_camera_distance

        #ToDo check error_pull_back is right!
        loss = euclidean_loss + self.weight_normal_loss* self.normal_point_loss()+error_pull_back
        if torch.isnan(loss):
            loss = error_pull_back


        if self.print_loss:
            print('Loss function : ', loss.cpu().detach().numpy(), 'Solution of camera pose optimization rotation: ',
                  self.r_param_rotation.cpu().detach().numpy(), 'dit: ', self.t_param_translation.cpu().detach().numpy(),
                  'Loss function distance : ', euclidean_loss.cpu().detach().numpy(),
                  'Loss function normal : ', self.weight_normal_loss * normal_loss.cpu().detach().numpy())
        return loss

    def pull_out_mesh(self, camera):
        #ToDo:plot camera and intersection point!
        #ToDo test without rotation!
        cam_center = camera.get_camera_center().clone()
        Tt = camera.T.clone()
        R_point_to_origin = look_at_rotation(camera_position = cam_center, at=((0, 0, 0),), up=((0, 1, 0), ), device= self.device)
        image_centre = np.array([[(self.inverse_renderer.image_size / 2), (self.inverse_renderer.image_size / 2)]]).astype(int)
        #image_centre = np.array([[(0), (0)]]).astype(int)
        #ToDo check correct T matrix, by plott!!

        T = torch.matmul(cam_center, R_point_to_origin.inverse())[0]

        camera.R = R_point_to_origin.inverse()
        #camera.T = Tt
## ToDo just plot the sanity check in the last iteration!!
        #self.plot_stitch_in_3D(camera, angle=120, save=False)

        intersection_point, _ = self.inverse_renderer(meshes_world=self.meshes.clone(),
                                                      node_pos=image_centre, R=R_point_to_origin, T=Tt, req_gradient=True, req_fragments=False)


        intersection_distance = (intersection_point.points_packed() ** 2).mean().sqrt()
        camera_distance = (camera.get_camera_center() ** 2).mean().sqrt()
        intersection_camera_distance = ((intersection_point.points_packed() - camera.get_camera_center()) ** 2).mean().sqrt()

        # print(intersection_distance.sqrt())
        # print(camera_distance.sqrt())
        # print(intersection_camera_distance.sqrt())
        return camera_distance, intersection_distance, intersection_camera_distance

    def euclidean_loss(self):
        euclidean_loss = ((self.world_point_cloud.points_packed()[self.matched_world]-self.current_point_cloud.points_packed()[self.matched_current])**2)  # ToDo increase the error function here if entries are nan!!!
        mask = torch.logical_not(torch.isnan(euclidean_loss))
        return euclidean_loss[mask].mean()

    def normal_point_loss(self):
        normal_loss = 1 - torch.cosine_similarity(self.world_point_cloud.normals_packed()[self.matched_world],
                                                  self.current_point_cloud.normals_packed()[self.matched_current], dim=1)
        mask = torch.logical_not(torch.isnan(normal_loss))
        return normal_loss[mask].mean()


    def plot_stitch_in_3D(self, camera, angle=0, save=False):
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
        with plt.style.context(('ggplot')):

            fig = plt.figure(figsize=(10, 7))
            ax = Axes3D(fig)

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



from models.multiview import CameraTorch, calc_reprojection_error_matrix, project_3d_points_to_image_plane_without_distortion
from pytorch3d.utils import opencv_from_cameras_projection
class ProjectionModel(nn.Module):
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
    def __init__(self, keypoints_3d, image_keypoints_2d, r_param_rotation, t_param_translation, camera, image_size, device, print_loss=False):
        super().__init__()
        mask = torch.logical_not(torch.isnan(keypoints_3d))[:, 0].cpu().detach()
        self.mesh_keypoints_3d = keypoints_3d[mask]
        self.image_keypoints_2d = torch.from_numpy(image_keypoints_2d)[mask].to(device=device)
        #self.match_brute_force = matches_brute_force
        self.device = device
        self.K = torch.eye(3).to(device=device)
        # Create an optimizable parameter for the x, y, z position of the camera.
        self.camera =camera
        self.image_size = image_size
        # Model Parameter:
        self.r_param_rotation = nn.Parameter(r_param_rotation.to(self.device))
        self.t_param_translation = nn.Parameter(t_param_translation.to(self.device))
        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=self.device)

        self.print_loss = print_loss

    def forward(self):
        # Render the image using the updated camera position. Based on the new position of the
        # camera we calculate the rotation and translation matrices
        R = self.cam_rodrigues_object.get_rotation_matrix(self.r_param_rotation)
        T = self.cam_rodrigues_object.get_translation_matrix(self.t_param_translation)
        self.camera.R = R
        self.camera.T = T
        R_cv, t_cv, K_cv = opencv_from_cameras_projection(cameras=self.camera, image_size=self.image_size[
            None].detach())  # ToDo check if K_cv makes sense (shift of principle points)
        cam_3d_ransac = CameraTorch(R_cv=R_cv, t_cv=t_cv, K=K_cv, image_size=self.image_size, device=self.device) #This camera is using the cv literature convention --> R,T,K needs to be converted!
        proj_matrix = cam_3d_ransac.projection
        re_projected_image_keypoints_2d =project_3d_points_to_image_plane_without_distortion(proj_matrix,self.mesh_keypoints_3d, image_size=self.image_size)
        #projected_image_keypoints_2d[:,1] =self.image_size[1]-projected_image_keypoints_2d[:,1]
        loss, error_residual = calc_reprojection_error_matrix(self.mesh_keypoints_3d, self.image_keypoints_2d, proj_matrix, image_size=self.image_size)

        # plt.figure()
        # print('loss in current iteration: ',loss)
        # i=2
        # img_cv = re_projected_image_keypoints_2d.clone().cpu().detach().int()
        # plt.plot(img_cv[:, 0], 128 - img_cv[:, 1], 'ro')
        # plt.plot(img_cv[i, 0], 128 - img_cv[i, 1], 'go')  # plot x and y using blue circle markers
        # img_p3d = self.image_keypoints_2d.clone().cpu().detach()
        # plt.plot(img_p3d[:, 0], 128 - img_p3d[:, 1], 'bo')
        # plt.plot(img_p3d[i, 0], 128 - img_p3d[i, 1], 'go')
        # plt.title('cv cam in red, pytorch cam in blue')
        # plt.show()

        return loss, error_residual, re_projected_image_keypoints_2d

    def evaluate(self):
        return self.r_param_rotation.detach(), self.t_param_translation.detach()

class ProjectionModelIntrinsicCalibration(nn.Module):
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
    def __init__(self, keypoints_3d, image_keypoints_2d, adjacency, r_rotation, t_translation, camera, image_size, device, print_loss=False):
        super().__init__()
        mask = torch.logical_not(torch.isnan(keypoints_3d))[:, 0].cpu().detach()
        self.mesh_keypoints_3d = keypoints_3d[mask]
        self.image_keypoints_2d = torch.from_numpy(image_keypoints_2d)[mask].to(device=device)
        self.adjacency = adjacency[mask][:, mask]
        #self.match_brute_force = matches_brute_force
        self.device = device
        self.K = torch.eye(3).to(device=device)
        self.camera = camera
        self.image_size = image_size
        self.focal_length = nn.Parameter(camera.focal_length.clone()) #ToDo: remove focal length as parameter! As soon all fitts!
        # self.principal_point_x = nn.Parameter(camera.get_principal_point()[0, 0].clone())
        # self.principal_point_y = nn.Parameter(camera.get_principal_point()[0, 1].clone())
        self.principal_point_x = camera.get_principal_point()[0, 0].clone()
        self.principal_point_y = camera.get_principal_point()[0, 1].clone()

        # Pose Parameter:
        self.r_param_rotation = r_rotation.to(self.device)
        self.t_param_translation = t_translation.to(self.device)
        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=self.device)

        self.print_loss = print_loss

    def forward(self):
        # Render the image using the updated camera position. Based on the new position of the
        # camera we calculate the rotation and translation matrices
        R = self.cam_rodrigues_object.get_rotation_matrix(self.r_param_rotation)
        T = self.cam_rodrigues_object.get_translation_matrix(self.t_param_translation)
        self.camera.R = R
        self.camera.T = T
        self.camera.focal_length = self.focal_length
        from pytorch3d.renderer.cameras import (
            PerspectiveCameras,
        )
        #ToDo: back to initial camera, new init is not needed!
        self.camera = PerspectiveCameras(device=self.device, R=R,
                                            T=T, focal_length=self.focal_length)

        R_cv, t_cv, K_cv = opencv_from_cameras_projection(cameras=self.camera, image_size=self.image_size[
            None].detach())  # ToDo check if K_cv makes sense (shift of principle points)

        # This transformation can be stated in words as:
        # “first scale, then rotate, then translate”.
        cam_3d_ransac = CameraTorch(R_cv=R_cv, t_cv=t_cv, K=K_cv, image_size=self.image_size,
                                    device=self.device)
        # This camera is using the cv literature convention --> R,T,K needs to be converted!
        proj_matrix = cam_3d_ransac.projection
        projected_image_keypoints_2d = project_3d_points_to_image_plane_without_distortion(proj_matrix,
                                                                                           self.mesh_keypoints_3d,
                                                                                           image_size=self.image_size,
                                                                                           convert_back_to_euclidean=True)
        fig = plot_graph(self.image_size.cpu().detach().numpy(), projected_image_keypoints_2d.cpu().detach().numpy(),  self.adjacency)
        plt.figure()
        img_cv = projected_image_keypoints_2d.clone().cpu().detach().int()
        plt.plot(img_cv[:, 0], 128-img_cv[:, 1], 'ro')  # plot x and y using blue circle markers
        img_p3d = self.image_keypoints_2d.clone().cpu().detach()
        plt.plot(img_p3d[:, 0], 128-img_p3d[:, 1], 'bo')
        plt.title('cv cam in red, pytorch cam in blue')
        plt.show()
        fig = plot_graph(self.image_size.cpu().detach().numpy(), img_cv.cpu().detach().numpy(),  self.adjacency)
        fig = plot_graph(self.image_size.cpu().detach().numpy(),img_p3d.cpu().detach().numpy(), self.adjacency)

        loss, error_residual = calc_reprojection_error_matrix(self.mesh_keypoints_3d, self.image_keypoints_2d,
                                                              proj_matrix, image_size=self.image_size)

        return loss, error_residual

    def evaluate(self):
        return self.focal_length, self.principal_point_x, self.principal_point_y

def camera_projection( keypoints_3d, camera, image_size, device, r_rotation=None, t_translation=None):
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
    if r_rotation is not None and t_translation is not None:
        cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=device)
        R = cam_rodrigues_object.get_rotation_matrix(r_rotation)
        T = cam_rodrigues_object.get_translation_matrix(t_translation)
        camera.R = R
        camera.T = T

    R_cv, t_cv, K_cv = opencv_from_cameras_projection(cameras=camera, image_size=image_size[None].detach())  # ToDo check if K_cv makes sense (shift of principle points)

    cam_3d_ransac = CameraTorch(R_cv=R_cv, t_cv=t_cv, K=K_cv, image_size=image_size, device=device)
    # This camera is using the cv literature convention --> R,T,K needs to be converted!
    proj_matrix = cam_3d_ransac.projection
    projected_image_keypoints_2d = project_3d_points_to_image_plane_without_distortion(proj_matrix, keypoints_3d, image_size=image_size, convert_back_to_euclidean=True)
    return projected_image_keypoints_2d


def euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous

    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)],
                         dim=1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")

def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def plot_img_2d_points(img_points_int, img_points_float, do_plot:bool =False, title:str='cv cam in red, pytorch cam in blue'):
    if isinstance(img_points_int, torch.Tensor):
        img_cv = img_points_int.clone().cpu().detach()
    if isinstance(img_points_float, torch.Tensor):
        img_p3d = img_points_float.clone().cpu().detach()

    if do_plot:
        plt.figure()
        plt.plot(img_cv[:, 0], 128 - img_cv[:, 1], 'ro')  # plot x and y using blue circle markers
        plt.plot(img_p3d[:, 0], 128 - img_p3d[:, 1], 'bo')
        plt.title(title)
        plt.show()

# def plot_nodes_on_img(image: np.ndarray, pos1: np.ndarray, pos2: np.ndarray=None, node_thick: int=1):
#     img = image.copy()
#     print(img)
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
#     # positions = pos.astype(int)
#     positions = pos1
#     for i in range(len(positions)):
#         cv2.circle(img, (positions[i][0], positions[i][1]), 0, (255, 0, 0), node_thick)
#
#     if pos2 is not None:
#         positions = pos2
#         for i in range(len(positions)):
#             cv2.circle(img, (positions[i][0], positions[i][1]), 0, (0, 255, 0), node_thick)
#
#     y_lim = img.shape[0]
#     x_lim = img.shape[1]
#     extent = 0, x_lim, 0, y_lim
#     fig = plt.figure(frameon=False, figsize=(20, 20))
#     plt.imshow(img, extent=extent, interpolation='nearest')
#     plt.show()
#     return fig
