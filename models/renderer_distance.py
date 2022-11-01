import os
import torch
from typing import Optional

from typing import NamedTuple, Sequence, Union
# datastructures
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex, SoftPhongShader, TensorProperties
)
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from utils.utils_sampel_points import _sample_points_from_meshes, _phong_sample_points_from_mesh
from utils.utils_sampel_points import _flip_sort_of_column, _rotate, _flip
from torch import nn


class   MeshRendererWithFragments2PointCloud(MeshRendererWithFragments):

    def __init__(self, rasterizer, shader) -> None:
        super().__init__(rasterizer, shader)
        self.node_fragments = None
        self.nodes2face_position = None
        self.barycentric_coords_position = None
        self.image_size = rasterizer.raster_settings.image_size
        self.current_node_pos = None
        self.blur_radius = rasterizer.raster_settings.blur_radius
        self.faces_per_pixel = rasterizer.raster_settings.faces_per_pixel


    def forward(self, meshes_world, node_pos, req_gradient=True, **kwargs):
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.
        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        For this set rasterizer.raster_settings.clip_barycentric_coords=True
        """
        if req_gradient:
            self.rasterizer.raster_settings.blur_radius = self.blur_radius
            self.rasterizer.raster_settings.faces_per_pixel= self.faces_per_pixel
        else:
            self.rasterizer.raster_settings.blur_radius = 0.0
            self.rasterizer.raster_settings.faces_per_pixel = 1

        fragments = self.rasterizer(meshes_world, **kwargs)

        #node_pos_fl = node_pos
        #rotated_node_pos = _flip(node_pos_fl)

        self.current_node_pos = node_pos
        node_fragments = self.stitch_graph_on_surface(fragments=fragments,node_pos =self.current_node_pos)
        k_dimensional_point_cloud, k_dimensional_point_normals = self.point_postions_from_surface_coords(meshes_world, node_fragments)
        merged_points = self.shader(node_fragments , k_dimensional_point_cloud, **kwargs)
        merged_normals = self.shader(node_fragments , k_dimensional_point_normals, **kwargs)
        features = torch.full(merged_points[..., :3].shape, 0.9, device=meshes_world.device)
        point_cloud = Pointclouds(points=merged_points[..., :3], normals=merged_normals[..., :3], features=features)
        return point_cloud

    def stitch_graph_on_surface(self, fragments: torch.Tensor, node_pos):
        # fragments:
        #   The outputs of rasterization. From this we use
        # - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
        #   of the faces (in the packed representation) which overlap each pixel in the image.
        # - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
        #   expand tensors with dummy dimensions is by inserting None into the axis we want to add
        nodes2face = fragments.pix_to_face[:, node_pos[:, 0], node_pos[:, 1]] #ToDo ensure that node_pos is always in range of image (default 128)
        nodes2face = nodes2face[:, None]
        zbuf = fragments.zbuf[:, node_pos[:, 0], node_pos[:, 1]]
        zbuf = zbuf[:, None]
        barycentric_coords = fragments.bary_coords[:,  node_pos[:, 0], node_pos[:, 1]]
        barycentric_coords = barycentric_coords[:, None]
        dists = fragments.dists[:, node_pos[:, 0], node_pos[:, 1]]
        dists = dists[:, None]
        node_fragments = Fragments(pix_to_face= nodes2face, zbuf=zbuf, bary_coords=barycentric_coords, dists=dists)
        return node_fragments

    def point_postions_from_surface_coords(self,meshes,node_fragments):
        point_cloud, point_normals = _phong_sample_points_from_mesh(meshes, node_fragments, return_normals=True)
        features = torch.full(point_cloud.shape, 0.9, device=meshes.device)
        return point_cloud, point_normals


class   SoftMesh2Image2PointCloudShader(nn.Module):
    """
    Calculate the merged point features by blending the top K faces for each pixel based
    on the 2d euclidean distance of the center of the pixel to the mesh face.

    Use this shader for generating silhouettes similar to SoftRasterizer [0].
    .. note::
        To be consistent with SoftRasterizer, initialize the
        RasterizationSettings for the rasterizer with
        `blur_radius = np.log(1. / 1e-4 - 1.) * blend_params.sigma`
    """
    def __init__(self, cameras: Optional[TensorProperties] = None, blend_params: Optional[BlendParams] = None) -> None:
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.cameras = cameras

    def forward(self, fragments: Fragments, points: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Only want to render the silhouette so RGB values can be ones.
        There is no need for lighting or texturing
        """
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                  or in the forward pass of SoftGouraudShader"
            raise ValueError(msg)
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        blend_params = kwargs.get("blend_params", self.blend_params)
        merged_points = sigmoid_3dporjection_blend(
            points, fragments, blend_params, znear=znear, zfar=zfar)
        return merged_points

def sigmoid_3dporjection_blend(
    points: torch.Tensor,
    fragments,
    blend_params: BlendParams,
    znear: Union[float, torch.Tensor] = 1.0,
    zfar: Union[float, torch.Tensor] = 100,
    ) -> torch.Tensor:
    """
    Args:
        points: (N, H, W, K, 3) three-dimensional point for each of the top K faces per pixel.
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction

    Returns:
        3dpoints+A (with alpha channel) merged_points: (N, NrNodes, 4)
    """
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    merged_points = torch.ones((N, H, W, 4), dtype=points.dtype, device=points.device)
    eps = 1e-10
    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0
    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(fragments.dists / blend_params.sigma) * mask
    #prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)
    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.
    # Reshape to be compatible with (N, H, W, K) values in fragments
    if torch.is_tensor(zfar):
        zfar = zfar[:, None, None, None]
    if torch.is_tensor(znear):
        znear = znear[:, None, None, None]
    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)
    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None]
    weighted_points = (weights_num[..., None] * points).sum(dim=-2)
    #weighted_background = delta * background
    merged_points[..., :3] = weighted_points/ denom
    merged_points[..., 3] = 1.0 - alpha

    return merged_points[:, 0]

