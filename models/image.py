import numpy as np
import torch
import torch.nn as nn
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
import copy
from models.pose_rodrigues_rot_fromula import Camera3DPoseRodriguesRotFormula
from tools_generate.NodeContainer import NodeContainer
from tools_generate.PolyGraph import PolyGraph
from tools_generate.mask import generate_mask

class Image(nn.Module):
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
    def __init__(self, node_pos: np.ndarray, nx_graph: PolyGraph, node_container: NodeContainer,  adjaceny, r_rotation, t_translation, batch_nr, item_in_batch, device,
                 attributes_orb = None, attributes = None,  adjacency_attributes =None, image_size = 128):
        super().__init__()
        #node_pos_fl = _flip_sort_of_column(node_pos=node_pos, image_size=image_size)
        #rotated_node_pos = _rotate(node_pos_fl, 90)
        self.node_pos = copy.deepcopy(node_pos)
        self.device = device
        self.adj = copy.deepcopy(adjaceny)
        self.attributes = copy.deepcopy(attributes)
        self.nx_graph = nx_graph
        self.node_container = node_container
        self.node_types = np.asarray(node_container.node_types)
        self.crossing_nodes_indices = self.node_types == 1
        self.end_nodes_indices = self.node_types == 2
        self.border_nodes_indices = self.node_types == 3
        self.type_indices = np.stack((self.crossing_nodes_indices,
                                      self.end_nodes_indices,
                                      self.border_nodes_indices), axis=-1)
        self.crossing_and_end_nodes_indices = np.logical_or(self.crossing_nodes_indices, self.end_nodes_indices)

        self.node_pos_of_crossing_and_end_nodes = self.node_pos[self.crossing_and_end_nodes_indices]
        self.end_node_indices_by_removed_border = self.end_nodes_indices[np.logical_not(self.border_nodes_indices)]
        self.crossing_node_indices_by_removed_border = self.crossing_nodes_indices[np.logical_not(self.border_nodes_indices)]

        self.remove_node_type = None
        self.adjacency_attributes =  copy.deepcopy(adjacency_attributes)
        #ToDo: delete orb features, since they are not really needed here!!
        #ToDo: this here should be a light weight class, since it gets saved for all seen images
        self.attributes_orb = torch.tensor(attributes_orb, dtype= torch.bool, device ='cpu') #needs to be on the cpu for the matching algorithm
        self.nodes2face = None
        self.barycentric_coords = None
        self.camera_view = None
        self.point_cloud = None
        self.image_size = image_size
        self.cam_rodrigues_object = Camera3DPoseRodriguesRotFormula(N=1, with_cam_mask=False, device=self.device)

        if r_rotation is not None and t_translation is not None:
            self.r_rotation = r_rotation.clone()
            self.t_translation = t_translation.clone()
            R_init = self.cam_rodrigues_object.get_rotation_matrix(self.r_rotation)
            T_init = self.cam_rodrigues_object.get_translation_matrix(self.t_translation)
            self.R = R_init
            self.T = T_init
        else:
            self.r_rotation = None
            self.t_translation = None
            self.R = None
            self.T = None
        self.pc = None
        self.batch_nr = copy.deepcopy(batch_nr)
        self.item_in_batch = copy.deepcopy(item_in_batch)
        self.is_image_of_interest = True
        self.matches_to_world_map = True

    def forward(self):
        print('Forward function of class Image needs to be implemented')
        return True

    def update_cam_position(self, r_rotation, t_translation):
        self.r_rotation = r_rotation.clone()
        self.t_translation = t_translation.clone()
        R_update, T_update = self.cam_rodrigues_object.get_rotation_and_translation(self.r_rotation, self.t_translation)
        self.R = R_update
        self.T = T_update

    def set_image_of_interest(self):
        self.is_image_of_interest = True

    def unset_image_of_interest(self):
        self.is_image_of_interest = False

    def get_orb_attributes(self):
        return self.attributes_orb

    def generate_mask_for_image(self, return_as_torch=False):
        mask = generate_mask(node_pos=self.node_pos,
                         mask_size=(self.image_size, self.image_size),
                         return_as_torch=return_as_torch, device=self.device)
        return mask

    def generate_matching_mask_for_image(self, return_as_torch=False):
        # make sure that mask is already set!
        mask = generate_mask(node_pos=self.node_pos[self.matches_to_world_map[:, 1]],
                         mask_size=(self.image_size, self.image_size),
                         return_as_torch=return_as_torch, device=self.device)
        return mask

    def render_point_could(self, points_per_pixel=1, radius=0.003, znear=1):
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
            compositor=AlphaCompositor(background_color=(0.2, 0.8, 0.3))
        )
        image_of_point_cloud = renderer(self.point_cloud)
        return image_of_point_cloud

    @property
    def get_remove_node_type(self):
        return self.remove_node_type

    @get_remove_node_type.setter
    def set_remove_node_type(self, value):
        self.remove_node_type = value

    @property
    def get_poly_graph(self):
        return self.nx_graph

    @property
    def get_node_container(self):
        return np.asarray(self.node_container)

    @property
    def get_adjacency(self):
        return np.asarray(self.nx_graph.adjacency())

    @property
    def get_node_pos(self):
        return np.asarray(self.nx_graph.positions_vector)

    @property
    def get_node_pos_of_crossing_and_end_nodes(self):
        return self.node_pos_of_crossing_and_end_nodes

    @property
    def get_crossing_and_end_nodes_indices(self):
        return self.crossing_and_end_nodes_indices

    @property
    def get_all_image_nodes(self):
        return self.node_pos

    @property
    def get_crossing_node_indices_by_removed_border(self):
        return self.crossing_node_indices_by_removed_border

    @property
    def get_end_node_indices_by_removed_border(self):
        return self.end_node_indices_by_removed_border

    @property
    def get_node_types(self):
        return np.asarray(self.node_types)

    @property
    def get_crossing_nodes_indices(self):
        return self.crossing_nodes_indices

    @property
    def get_end_nodes_indices(self):
        return self.end_nodes_indices

    @property
    def get_border_nodes_indices(self):
        return self.border_nodes_indices

    @property
    def get_type_indices(self):
        return self.type_indices

    @property
    def get_stacked_adj_matrix(self):
        return np.asarray(self.nx_graph.squared_stacked_adj_matrix())

    @property
    def get_Rotation_and_Translation(self):
        if self.R is not None and self.T is not None:
            return self.R, self.T
        else:
            print('update all position parameters first! Therefore call: .update_cam_position()')

    @property
    def get_rotation_vector(self):
        return self.r_rotation

    @property
    def get_translation_vector(self):
        return self.t_translation

    @property
    def get_matches_to_world_map(self):
        return self.matches_to_world_map

    def set_matches_with_world_map(self, matches):
        assert self.matches_to_world_map # to ensure that the match is written just once! works and is important
        # however it is annyoing for debugging
        self.matches_to_world_map = matches


