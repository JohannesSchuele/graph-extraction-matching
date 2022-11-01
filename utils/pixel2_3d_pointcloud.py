import os
import torch
import matplotlib.pyplot as plt

from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm.notebook import tqdm

import sys
from typing import Tuple, Union

import torch
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments

from pytorch3d.structures import Pointclouds
import torch.nn as nn
from pytorch3d.renderer.cameras import FoVOrthographicCameras
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    FoVPerspectiveCameras
)
from mpl_toolkits.mplot3d import Axes3D

class GraphImage(nn.Module):
    """
    Collect all data and operations that are related to an image extracted graph.
    init:
         face_attributes: packed attributes of shape (total_faces, 3, D),
            specifying the value of the attribute for each
            vertex in the face.
    Returns:
        pixel_vals: tensor of shape (N, H, W, K, D) giving the interpolated
        value of the face attribute for each pixel.
        textures = new_src_mesh.textures.sample_textures(fragments,new_src_mesh.faces_packed())
    """
    def __init__(self,node_pos: np.ndarray, adj, attributes: torch.Tensor,image_size =128 ):
        super().__init__()
        node_pos_fl = _flip_sort_of_column(node_pos=node_pos, image_size=image_size)
        rotated_node_pos = _rotate(node_pos_fl, 90)
        self.node_pos = rotated_node_pos
        self.adj = adj
        self.attributes = attributes
        self.nodes2face = None
        self.barycentric_coords = None
        self.camera_view = None
        self.point_cloud = None
        self.image_size = image_size
        self.R = None
        self.T = None
        self.pc = None

    def forward(self,fragments: torch.Tensor, meshes = None, R= None, T=None):
        self.stitch_graph_on_surface(fragments, R= R, T=T)
        self.surface_coord_2_point_cloud(meshes=meshes)
        return self.point_cloud


    def stitch_graph_on_surface(self,fragments: torch.Tensor, camera_view = None, R= None, T=None):
        # fragments:
        #   The outputs of rasterization. From this we use
        # - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
        #   of the faces (in the packed representation) which overlap each pixel in the image.
        # - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
        #   the barycentric coordinates of each pixel relative to the faces (in the packed representation) which overlap the pixel.
        # ToDo check that the image is mapped on the surface!! if -1 --> LoopClosure or increase the model (or put the camerea closer to the model!)
        self.nodes2face = fragments.pix_to_face[:, self.node_pos[:,0], self.node_pos[:,1], 0]#ToDO test if first best (KNN) works well enough
        self.barycentric_coords = fragments.bary_coords[:,  self.node_pos[:,0], self.node_pos[:,1], 0]
        if camera_view is not None:
            self.camera_view = camera_view
            self.R = camera_view.R
            self.T = camera_view.T
        else:
            self.R = R
            self.T = T


    def surface_coord_2_point_cloud(self,meshes):
        point_cloud, point_normals = _sample_points_from_meshes(meshes,self.nodes2face,self.barycentric_coords,return_normals= True, return_textures= False,)
        # features:
        #     Can be either
        #     - List where each element is a tensor of shape (num_points, C)
        #       containing the features for the points in the cloud.
        #     - Padded float tensor of shape (num_clouds, num_points, C).
        #     where C is the number of channels in the features.
        #     For example 3 for RGB color.
        self.pc =point_cloud
        features = torch.full(point_cloud.shape, 0.9, device=meshes.device)
        self.point_cloud = Pointclouds(points=point_cloud,  normals=point_normals, features=features)





    def render_point_could(self, points_per_pixel=1, radius=0.003, znear=1):
        # camera_4_point_cloud = FoVOrthographicCameras(device=self.camera_view.device, R=self.camera_view.R, T=self.camera_view.T, znear=znear)
        camera_4_point_cloud =FoVPerspectiveCameras(device=self.camera_view.device, R=self.R, T=self.T,znear=znear)
        camera_4_point_cloud = self.camera_view
        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 128x128. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size,
            radius= radius,
            points_per_pixel= points_per_pixel
        )
        # Create a points renderer by compositing points using an alpha compositor (nearer points
        # are weighted more heavily). See [1] for an explanation.
        rasterizer = PointsRasterizer(cameras=camera_4_point_cloud , raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            # Pass in background_color to the alpha compositor, setting the background color
            # to the 3 item tuple, representing rgb on a scale of 0 -> 1, in this case blue
            compositor=AlphaCompositor(background_color=(0.2,0.8, 0.3))
        )
        image_of_point_cloud = renderer(self.point_cloud)
        return image_of_point_cloud


    def plot_pointcloud(self, title="Point Cloud"):
        # Sample points uniformly from the surface of the mesh.
        points = self.point_cloud.points_packed()
        x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
        fig = plt.figure(figsize=(10,10))
        ax = Axes3D(fig)
        #fig.add_axes(ax)
        ax.scatter3D(x, z, -y)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_title(title)
        #ax.view_init(0, 0)
        plt.show()


# def plot_pointcloud(mesh, title=""):
#     # Sample points uniformly from the surface of the mesh.
#     points = sample_points_from_meshes(mesh, 5000)
#     x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
#     fig = plt.figure(figsize=(5, 5))
#     ax = Axes3D(fig)
#     ax.scatter3D(x, z, -y)
#     ax.set_xlabel('x')
#     ax.set_ylabel('z')
#     ax.set_zlabel('y')
#     ax.set_title(title)
#     ax.view_init(190, 30)
#     plt.show()


def _sample_points_from_meshes(
    meshes,
    face_verts_GraphNode,
    barycentric_graph,
    return_normals: bool = False,
    return_textures: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    #face_verts_GraphNode (N, num_samples)
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.

    Returns:
        3-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
          For empty meshes the corresponding row in the normals array will
          be filled with 0.
        - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          For empty meshes the corresponding row in the textures array will
          be filled with 0.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """
    num_samples = face_verts_GraphNode.shape[1]
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    if return_textures and meshes.textures is None:
        raise ValueError("Meshes do not contain textures.")

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(
            areas, mesh_to_face[meshes.valid], max_faces
        )  # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        # sample_face_idxs = areas_padded.multinomial(
        #     num_samples, replacement=True
        # )  # (N, num_samples)
        # sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    sample_face_idxs = face_verts_GraphNode

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0 = barycentric_graph[:, :, 0]
    w1 = barycentric_graph[:, :, 1]
    w2 = barycentric_graph[:, :, 2]


    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c


    #samples[meshes.valid] = a

    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_face_idxs]
        normals[meshes.valid] = vert_normals

    ## texture may needs to be adapted
    if return_textures:
        # fragment data are of shape NxHxWxK. Here H=S, W=1 & K=1.
        pix_to_face = face_verts_GraphNode.view(len(meshes), num_samples, 1, 1)  # NxSx1x1
        bary = torch.stack((w0, w1, w2), dim=2).unsqueeze(2).unsqueeze(2)  # NxSx1x1x3
        # zbuf and dists are not used in `sample_textures` so we initialize them with dummy
        dummy = torch.zeros(
            (len(meshes), num_samples, 1, 1), device=meshes.device, dtype=torch.float32
        )  # NxSx1x1
        fragments = MeshFragments(
            pix_to_face=pix_to_face, zbuf=dummy, bary_coords=bary, dists=dummy
        )
        textures = meshes.sample_textures(fragments)  # NxSx1x1xC
        textures = textures[:, :, 0, 0, :]  # NxSxC

    # return
    # ToDo (gkioxari) consider returning a Pointclouds instance [breaking]
    if return_normals and return_textures:
        # pyre-fixme[61]: `normals` may not be initialized here.
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, normals, textures
    if return_normals:  # return_textures is False
        # pyre-fixme[61]: `normals` may not be initialized here.
        return samples, normals
    if return_textures:  # return_normals is False
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, textures
    return samples


def _rotate(vector, theta, rotation_around=None) -> np.ndarray:
    """
    reference: https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    :param vector: list of length 2 OR
                   list of list where inner list has size 2 OR
                   1D numpy array of length 2 OR
                   2D numpy array of size (number of points, 2)
    :param theta: rotation angle in degree (+ve value of anti-clockwise rotation)
    :param rotation_around: "vector" will be rotated around this point,
                    otherwise [0, 0] will be considered as rotation axis
    :return: rotated "vector" about "theta" degree around rotation
             axis "rotation_around" numpy array
    """
    vector = np.array(vector)
    if vector.ndim == 1:
        vector = vector[np.newaxis, :]
    if rotation_around is not None:
        vector = vector - rotation_around
    vector = vector.T
    theta = np.radians(theta)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    output: np.ndarray = (rotation_matrix @ vector).T
    if rotation_around is not None:
        output = output + rotation_around
    return output.squeeze()


def _flip_sort_of_column(node_pos, image_size):
    resorted_node_pos = node_pos
    for i in range(node_pos.shape[0]):
        resorted_node_pos[i, :] = [node_pos[i][0], image_size - node_pos[i][1]]
    return resorted_node_pos