import numpy as np
import torch
from models.pc_alignment_pose_opt import scale_point_cloud


# # This is an idea to improve management of different data (ojects including
# # point cloud however no yet used because of time
# class DMDContainer():
#     """
#     Container object for storing DepthMeasurementData objects and relevant
#     variables
#     """
#     def __init__(self, synthetic_camera):
#         self.data_list = []
#         self.feature_pixel_pairs = None
#
#
#     def add_depth_map_data(self, new_depth_map_data):
#         self.data_list += new_depth_map_data
#
#     @property
#     def dmd(self):
#         return self.data_list


class DepthMeasurementData():
    """
    This class manages different objects that are in connection with the depth
    map measurements and are important for the deformation optimization
    """

    def __init__(self, measurement_depth_map, measurement_image, device,
                 measurement_point_cloud=None,
                 measurement_key_point_cloud=None, measurement_node_pos_key=None,
                 synthetic_camera=None,
                 point_cloud_alignment_T=None,
                 point_cloud_alignment_R=None,
                 point_cloud_alignment_xy_scale=None,
                 point_cloud_scale_z=None,
                 ):
        """

        Args:
            measurement_depth_map:
            measurement_image:
            measurement_point_cloud:
            measurement_key_point_cloud:
            measurement_node_pos_key:
            synthetic_camera:
            point_cloud_alignment_R:
            point_cloud_alignment_T:
            point_cloud_alignment_xy_scale:
        """
        self.measurement_depth_map = measurement_depth_map
        self.measurement_image = measurement_image
        self.device = device
        self.measurement_point_cloud = measurement_point_cloud
        self.measurement_key_point_cloud = measurement_key_point_cloud
        self.measurement_node_pos_key = measurement_node_pos_key
        self._synthetic_camera = synthetic_camera

        self.point_cloud_alignment_R = point_cloud_alignment_R
        self.point_cloud_alignment_T = point_cloud_alignment_T
        self.point_cloud_alignment_xy_scale = point_cloud_alignment_xy_scale

        self._image_feature_pairs_positions = None
        self._contour_image_pair = None


    def set_key_point_cloud(self, measurement_key_point_cloud, node_pos_key):
        self.measurement_key_point_cloud = measurement_key_point_cloud
        self.measurement_node_pos_key = node_pos_key

    @property
    def torch_measurement_image(self):
        torch_measurement_image = torch.from_numpy(self.measurement_image)
        return torch_measurement_image.cuda()

    @property
    def synthetic_camera(self):
        if self._synthetic_camera is None:
            print('Synthetic camera not defined!!!')
        return self._synthetic_camera

    @synthetic_camera.setter
    def synthetic_camera(self, synthetic_camera):
        self._synthetic_camera = synthetic_camera

    def set_point_cloud_alignment(self, R=None, T=None, xy_scale=None):
        if R is not None:
            self.point_cloud_alignment_R = R
        if T is not None:
            self.point_cloud_alignment_T = T
        if xy_scale is not None:
            self.point_cloud_alignment_xy_scale = xy_scale

    def get_point_cloud_alignment(self):
        return self.point_cloud_alignment_R, self.point_cloud_alignment_T, self.point_cloud_alignment_xy_scale

    @property
    def initial_point_cloud_alignment(self):
        """
        Returns: Initial position of point cloud for preparation before alignment on mesh. Use point cloud alignment
         values R and T if already specified, else use default best guess values.
        """
        out_T = self.point_cloud_alignment_T if self.point_cloud_alignment_T is not None \
            else torch.tensor([0, 0, 1], device=self.device, dtype=torch.float)
        camera_rotation = self._synthetic_camera.R.to(torch.float) if self._synthetic_camera is not None else\
            torch.eye(3, dtype=torch.float, device=self.device)
        out_R = self.point_cloud_alignment_R if self.point_cloud_alignment_R is not None else camera_rotation
        return out_R, out_T


    @property
    def scaled_measurement_point_cloud(self):
        if self.point_cloud_alignment_xy_scale is None:
            print("! No point cloud scale parameter set")
            return scale_point_cloud(self.measurement_point_cloud, scale_xy_plane=1)
        return scale_point_cloud(self.measurement_point_cloud, scale_xy_plane=self.point_cloud_alignment_xy_scale)

    # Todo: Pair is saved only in container of one dataset -> the information might be incorrectly used because only
    #       the second image is linked to the feature position info -> maybe create object to manage these data
    #       container objets and save the info ther in conjunction with mechanisms to ensure correct use
    def set_image_feature_pairs_positions(self, a_positions, b_positions, overwrite=True):
        if overwrite or self._image_feature_pairs_positions is None:
            self._image_feature_pairs_positions = (a_positions, b_positions)
        else:
            a, b = self._image_feature_pairs_positions
            self._image_feature_pairs_positions = (np.concatenate((a, a_positions), axis=0),
                                                   np.concatenate((b, b_positions), axis=0))

    @property
    def image_feature_pairs_positions(self):
        return self._image_feature_pairs_positions

    def set_contour_image_pair(self, contour_a, contour_b):
        self._contour_image_pair = contour_a, contour_b

    @property
    def contour_image_pair(self):
        if self._contour_image_pair is None:
            print('Warning: no contour images are loaded')
        return self._contour_image_pair
