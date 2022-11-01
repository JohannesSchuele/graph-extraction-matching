import cv2
import numpy as np
import pytorch3d.transforms
import torch
from numpy import genfromtxt
from configs.plot.config_plots import *
from configs.plot.colours import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from pytorch3d.transforms import axis_angle_to_matrix
from pathlib import Path
import json

from tools_generate.manage_measurement_data import DepthMeasurementData

class DataDepthMap():

    def __init__(self, camera_synthetic_init, dim=(128, 128), scale_xy_plane= 1.0,
                 scale_z_direction =1.0, move_x_from_center=-0.5, move_y_from_center_and_decrease=0.3,
                 centralize_torch_point_cloud=True, get_key_nodes =True, device='cpu', crop_data=True):
        self.call_nr = 0

        experiment = 'balloon_tattoo'
        database = {'balloon_tattoo': ["../../../cstaerk/data/depth_map/balloon_tattoo/", 'LB_20220927_001']}
        self.data_dr = Path(database[experiment][0])
        self.sample_id = database[experiment][1] # get_sample_list(self.data_dr, glob_pattern='*_cropped.tiff')

        self.camera_synthetic_init = camera_synthetic_init
        self.bbox = None
        self.dim = dim
        self.move_x_from_center = move_x_from_center
        self.move_y_from_center_and_decrease = move_y_from_center_and_decrease
        self.scale_xy_plane = scale_xy_plane
        self.scale_z_direction = scale_z_direction
        self.device = device
        self.centralize = centralize_torch_point_cloud
        self.get_key_nodes = get_key_nodes
        self.crop_data = crop_data

    def forward(self, get_torch_point_cloud=False, show_loaded_data=True, show_cropped_data=True, move_x_from_center=None, move_y_from_center_and_decrease=None):
        # get path of current sample by counting up letters in following list
        nr2alphabet = ['A', 'B']
        current_sample_fp = str(self.data_dr / (self.sample_id + nr2alphabet[self.call_nr])) #if self.call_nr <= len(nr2alphabet) else len(nr2alphabet)]))

        # load data
        depth_map_loaded = self.load_csv_image(fp=current_sample_fp)            # image
        surface_image_loaded = self.load_image(fp=current_sample_fp)/255        # depth map

        from tools_generate.image_processing import skeletonise_and_clean
        from utils.deformation_helpers import get_index_matrix, get_point_cloud

        if not self.call_nr == 0:  # not 1st data set - only for comparisons    # features - skeleton and manual
            previous_sample_fp = str(self.data_dr / (self.sample_id + nr2alphabet[self.call_nr - 1]))
            if Path(previous_sample_fp + '_WL_sceleton.csv').exists() \
                    and Path(current_sample_fp + '_WL_sceleton.csv').exists():
                # Load structure images, skeletonize and convert to point clouds
                def point_cloud_from_contour(sample_fp):
                    contour_image = self.load_csv_image(fp=sample_fp,
                                                        ending='_WL_sceleton.csv')
                    index_matrix = get_index_matrix(device=self.device, image_size_int=contour_image.shape)
                    contour_skeleton = skeletonise_and_clean(thr_image=contour_image,
                                                             plot=True, save=False, directory='')
                    img_torch = torch.stack((torch.tensor(contour_skeleton),    # 3 channel image needed - grey image
                                             torch.tensor(contour_skeleton),
                                             torch.tensor(contour_skeleton)),
                                            dim=2)[None].to(device=self.device)
                    point_cloud = get_point_cloud(img_torch=img_torch,
                                                  mask_torch=None,
                                                  threshold=0.5,
                                                  idx_matrix=index_matrix)
                    return point_cloud

                point_cloud_a = point_cloud_from_contour(previous_sample_fp)
                point_cloud_b = point_cloud_from_contour(current_sample_fp)
                contour_point_clouds = (point_cloud_a, point_cloud_b)
            else:
                contour_point_clouds = None, None

            match_data_loaded = self.load_match_data(fp=current_sample_fp) \
                if Path(current_sample_fp+'_match_data.json').exists() else None
        else:
            match_data_loaded, contour_point_clouds = None, None

        self.call_nr += 1

        # crop and resize
        self.determine_bbox(depth_map=depth_map_loaded, move_x_from_center=move_x_from_center,
                            move_y_from_center_and_decrease=move_y_from_center_and_decrease)

        depth_map_cropped = imcrop(img=depth_map_loaded, bbox=self.bbox)
        depth_map_resized = cv2.resize(depth_map_cropped, self.dim, interpolation=cv2.INTER_AREA)

        surface_image_cropped = imcrop(img=surface_image_loaded, bbox=self.bbox)
        surface_image_resized = cv2.resize(surface_image_cropped, self.dim, interpolation=cv2.INTER_AREA)

        if contour_point_clouds is not None:  # assuming img dim=contour img dim
            contour_images_resized = \
                crop_and_resize_point_cloud(contour_point_clouds[0], bbox=self.bbox,
                                            original_and_goal_dim=(surface_image_loaded.shape, self.dim)), \
                crop_and_resize_point_cloud(contour_point_clouds[1], bbox=self.bbox,
                                            original_and_goal_dim=(surface_image_loaded.shape, self.dim))
        else:
            contour_images_resized = None

        if match_data_loaded is not None:
            match_data_resized = crop_and_resize_match_data(match_data_loaded.copy(), bbox=self.bbox,
                                                            original_and_goal_dim=(depth_map_cropped.shape, self.dim))
        else:
            match_data_resized = None

        # only for testing:
        from tests.generate_test_data import arrange_example_depth_map, arrange_example_image
        surface_image_resized = arrange_example_image(self.dim)
        depth_map_resized = arrange_example_depth_map(self.dim, profile=(self.call_nr == 2))

        # plot
        if show_loaded_data:
            print('Plot original loaded depth map and surface image!')
            plot_data(depth_map_loaded, surface_image_loaded, match_data=match_data_loaded,
                      title=' original loaded data', call_nr=self.call_nr)

        if show_cropped_data:
            plot_data(depth_map_resized, surface_image_resized, match_data=match_data_resized,
                      title=' cropped and resized data', call_nr=self.call_nr)
            plot_depth_map_as_3dcloud(depth_map_resized, title=' cropped and resized data',
                                      call_nr=self.call_nr)

        print('Be carefull! origin in the left upper corner of the image \n'
              ' in image coordinate system x corresponds to columns and y- to rows!!')
        #Todo transform image plane!

        # object to manage data and point clouds in one
        depth_map_data_container = DepthMeasurementData(depth_map_resized, surface_image_resized, self.device)
        if contour_images_resized is not None:
            depth_map_data_container.set_contour_image_pair(*contour_images_resized)
        if match_data_resized is not None:
            depth_map_data_container.set_image_feature_pairs_positions(
                a_positions=match_data_resized['pointsA'],
                b_positions=match_data_resized['pointsB'])

        # (get point cloud of depth map (and key points of that depth map)) and return data
        if get_torch_point_cloud:
            torch_point_cloud = self.transform_depth_map_2_torch_point_cloud(depth_map=depth_map_resized, centralize=self.centralize)
            torch_point_cloud = self.scale_point_cloud(torch_point_cloud=torch_point_cloud)

            depth_map_data_container.measurement_point_cloud = torch_point_cloud

            if self.get_key_nodes:
                torch_key_point_cloud, node_pos_key = self.get_node_keys_with_pc(depth_map=depth_map_resized, centralize=self.centralize)
                torch_key_point_cloud = self.scale_point_cloud(torch_point_cloud=torch_key_point_cloud)
                if show_cropped_data:
                    plot_torch_3dtensor(torch_point_cloud_tensor=torch_point_cloud, torch_point_cloud_tensor_2=torch_key_point_cloud,
                                        title='Point cloud with key point cloud based on torch tensor!')
                    #node_pos_key = node_pos_key.flip(1)
                    node_pos_key_tmp = node_pos_key
                    # node_pos_key[0] = node_pos_key_tmp[1]
                    # node_pos_key_tmp[1] = node_pos_key_tmp[2]

                depth_map_data_container.set_key_point_cloud(torch_key_point_cloud, node_pos_key)

                return depth_map_data_container

            else:
                if show_cropped_data:
                    plot_torch_3dtensor(torch_point_cloud_tensor=torch_point_cloud,
                                        title='Point cloud based on torch tensor!')
                return depth_map_data_container
        else:
            return depth_map_data_container

    def transform_depth_map_2_torch_point_cloud(self, depth_map, centralize=True):
        # create empty tensor with shape 1x(number of pixels)x(3 dimensions)
        torch_point_cloud = torch.zeros((1, depth_map.shape[1]*depth_map.shape[0], 3),
                                        device=self.device, dtype=torch.float)
        dim_in_x = depth_map.shape[1]
        dim_in_y = depth_map.shape[0]
        for x_column_itr in range(dim_in_x):
            # in image coordinate system x corresponds to columns and y- to rows!!
            # (so coordinate system reproduces what you would see in image!)
            torch_point_cloud[0, range(x_column_itr*dim_in_y, (x_column_itr+1)*dim_in_y), :] = \
                torch.cat((torch.full((dim_in_x,), x_column_itr, device=self.device, dtype=torch.float)[:, None],  # x
                           torch.tensor(range(0, dim_in_y), device=self.device, dtype=torch.float)[:, None],       # y
                           torch.tensor(depth_map[range(0, dim_in_y), x_column_itr], device=self.device, dtype=torch.float)[:, None]
                           ), 1)

        #Tmp
        for x_column_itr in range(dim_in_x):
            # in image coordinate system x corresponds to columns and y- to rows!!
            # (so coordinate system reproduces what you would see in image!)

            torch_point_cloud[0, range(x_column_itr*dim_in_y, (x_column_itr+1)*dim_in_y), :] = \
                torch.cat((torch.full((dim_in_x,), x_column_itr, device=self.device, dtype=torch.float)[:, None],  # x
                           torch.tensor(reversed(range(0, dim_in_y)), device=self.device, dtype=torch.float)[:, None],       # y
                           torch.tensor(depth_map[reversed(range(0, dim_in_y)), x_column_itr], device=self.device, dtype=torch.float)[:, None]
                           ), 1)
        if centralize:
            torch_point_cloud[:, :, 0] = torch_point_cloud[:, :, 0]-dim_in_x/2
            torch_point_cloud[:, :, 1] = torch_point_cloud[:, :, 1]-dim_in_y/2

        torch_point_cloud_in_view_coordinates = point_cloud_orientation_correction(torch_point_cloud, self.camera_synthetic_init)

        return torch_point_cloud_in_view_coordinates.to(device=self.device)


    def get_node_keys_with_pc(self, depth_map, centralize=True):
        torch_key_point_cloud = torch.zeros((1, 5, 3), device=self.device, dtype=torch.float)
        dim_in_x = depth_map.shape[1]
        dim_in_y = depth_map.shape[0]
        node_pos = torch.tensor([[0, 0], [0, dim_in_x-1], [dim_in_y-1, 0], [dim_in_y-1, dim_in_x-1],
                                 [int(dim_in_y/2), int(dim_in_x/2)]], device='cpu', dtype=torch.long)

        node_pos = torch.tensor([[0, 0], [0, dim_in_x - 1], [dim_in_y - 1, 0], [dim_in_y - 1, dim_in_x - 1],
                                 [int(dim_in_y / 2), int(dim_in_x / 2)]], device='cpu', dtype=torch.long)



        # Todo check order of node_pos!!!
        torch_key_point_cloud[0, :, 2] = torch.from_numpy(depth_map[node_pos[:, 1], node_pos[:, 0]]).to(device=self.device) # z from dm at (H, W)
        torch_key_point_cloud[0, :, 0] = node_pos[:, 1].to(dtype=torch.float)        # x - W in (H, W)
        torch_key_point_cloud[0, :, 1] = node_pos[:, 0].to(dtype=torch.float)        # y - H in (H, W)



        if centralize:
            torch_key_point_cloud[:, :, 0] = torch_key_point_cloud[:, :, 0]-dim_in_x/2
            torch_key_point_cloud[:, :, 1] = torch_key_point_cloud[:, :, 1] - dim_in_y/2

        # torch_key_point_cloud_in_view_coordinates = point_cloud_orientation_correction(torch_key_point_cloud, self.camera_synthetic_init)

        return torch_key_point_cloud.to(device=self.device), node_pos


    def scale_point_cloud(self, torch_point_cloud, scale_xy_plane=None, scale_z_direction=None):
        if scale_xy_plane is None or scale_z_direction is None:
            torch_point_cloud[0, :, 0:2] = torch_point_cloud[0, :, 0:2]*self.scale_xy_plane
            torch_point_cloud[0, :, 2] = torch_point_cloud[0, :, 2]*self.scale_z_direction
        else:
            torch_point_cloud[0, :, 0:2] = torch_point_cloud[0, :, 0:2] * scale_xy_plane
            torch_point_cloud[0, :, 2] = torch_point_cloud[0, :, 2] * scale_z_direction

        return torch_point_cloud





    def load_csv_image(self, fp, ending='_depth.csv'):
        """
        Args:
            fp: file stem
            ending: to add to stem
        Returns: array with data
        """
        #target_point_cloud = genfromtxt('../data/DepthMap/SB_20220228_007_001_depth.csv')
        fp_point_cloud = fp+ending
        target_point_cloud = genfromtxt(fp_point_cloud)
        return target_point_cloud

    def load_image(self, fp):
        #fp_bladder_image = "../data/DepthMap/SB_20220228_007_001.tiff"
        fp_bladder_image = fp+'.tiff'
        bladder_image = cv2.imread(fp_bladder_image)
        bladder_image = cv2.cvtColor(bladder_image, cv2.COLOR_BGR2RGB)
        return bladder_image

    def load_match_data(self, fp):
        fp_match_data = fp + '_match_data.json'
        with open(fp_match_data, 'r') as file:
            match_data = json.load(file)
        return match_data

    def determine_bbox(self, depth_map, move_x_from_center, move_y_from_center_and_decrease):
        if self.crop_data:
            if self.bbox is None:
                if move_x_from_center is None or move_y_from_center_and_decrease is None:
                    self.bbox = get_centralized_bbox(depth_map, move_x_from_center=self.move_x_from_center,
                                                     move_y_from_center_and_decrease=self.move_y_from_center_and_decrease)
                else:
                    self.bbox = get_centralized_bbox(depth_map, move_x_from_center=move_x_from_center,
                                                     move_y_from_center_and_decrease=move_y_from_center_and_decrease)
            elif move_x_from_center is not None and move_y_from_center_and_decrease is not None:
                self.bbox = get_centralized_bbox(depth_map, move_x_from_center=move_x_from_center,
                                                 move_y_from_center_and_decrease=move_y_from_center_and_decrease)
        else:
            self.bbox = None  # 0, 0, depth_map.shape[0], depth_map.shape[1]

    def update_scale_xy_plane_parameter(self, scale_xy_plane_parameter):
        print('Scaling of point cloud now implemented in DepthMeasurementData '
              'Class! Check which one is used to avoid scaling twice')
        self.scale_xy_plane = self.scale_xy_plane*scale_xy_plane_parameter


def resort_matrix(matrix):
    resorted_matrix = np.flip(matrix, 0)
    resorted_matrix = np.flip(matrix, 1)
    return resorted_matrix

def plot_data(depth_map, surface_image, match_data=None, title:str = '', call_nr=None):
    print('Plot original loaded depth map and surface image!')
    fig_depth_map = plt.figure(figsize=(10, 10))
    plt.imshow(depth_map)
    plt.title('Depth map of ' + title)
    plt.axis("off")
    # plt.show()
    save_figure(fig_depth_map, name_of_figure='cropped_depth_map_data' + str(call_nr), show_fig=True)

    fig_surface_image = plt.figure(figsize=(10, 10))
    plt.imshow(surface_image)
    if match_data is not None:  # plot loaded feature points, here still in format [x, y]
        pos1, pos2 = np.array(match_data['pointsA']), np.array(match_data['pointsB'])
        plt.scatter(x=pos1[:, 0], y=pos1[:, 1], marker='x', s=5, c=blue_2, alpha=0.9)
        plt.scatter(x=pos2[:, 0], y=pos2[:, 1], marker='x', s=5, c=red_1, alpha=0.9)
        for p1, p2 in zip(pos1, pos2):
            plt.plot((p1[0], p2[0]), (p1[1], p2[1]))
    plt.title('Surface image of '+title)
    plt.axis("off");
    # plt.show()
    save_figure(fig_surface_image, name_of_figure='cropped_image_data' + str(call_nr), show_fig=True)


def plot_depth_map_as_3dcloud(depth_map, title:str ='', call_nr=None):
    with plt.style.context(('ggplot')):
        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)
        for x_row in range(depth_map.shape[1]):
            # in image coordinate system x corresponds to columns and y- to rows!!
            # (so coordinate system reproduces what you would see in image!)
            ax.scatter(xs=float(x_row), ys=range(depth_map.shape[0]), zs=depth_map[:, x_row],
                       color=green_6, alpha=0.8, s=2)
        ax.set_title('Scatter plot of depth map'+title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # fig.show()
        save_figure(fig, name_of_figure='depth_map_point_cloud'+str(call_nr), show_fig=True)

def plot_torch_3dtensor(torch_point_cloud_tensor, torch_point_cloud_tensor_2 =None , title:str ='', plot_view_position=None):
    if torch_point_cloud_tensor.dim() == 3:
        point_cloud_tensor = torch_point_cloud_tensor[0].cpu().detach().numpy()
    else: point_cloud_tensor = torch_point_cloud_tensor.cpu().detach().numpy()
    with plt.style.context(('ggplot')):
        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)
        ax.scatter(xs=point_cloud_tensor[:, 0], ys=point_cloud_tensor[:, 1], zs=point_cloud_tensor[:, 2],
                       color=green_6, alpha=0.7, s=1)
        if torch_point_cloud_tensor_2 is not None:
            torch_point_cloud_tensor_2 = torch_point_cloud_tensor_2[0].cpu().detach().numpy()
            ax.scatter(xs=torch_point_cloud_tensor_2[:, 0], ys=torch_point_cloud_tensor_2[:, 1], zs=torch_point_cloud_tensor_2[:, 2],
                       color=red_3, alpha=1, s=5)
        ax.set_title('Scatter plot of pytorch tensor'+title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if plot_view_position is not None:
            ax.view_init(plot_view_position[0], plot_view_position[1])
        fig.show()

def get_centralized_bbox(init_image, move_y_from_center_and_decrease=0.0, move_x_from_center=0.0):
    if init_image.shape[0] <= init_image.shape[1]:

        half_diff_y = init_image.shape[0]/2
        # in image coordinate system x corresponds to columns and y- to rows!!
        # origin in the left upper corner of the image!!
        # (so coordinate system reproduces what you would see in image!)
        if move_y_from_center_and_decrease >= 0:
            y1 = int(half_diff_y*move_y_from_center_and_decrease)
            y2 = init_image.shape[0]
        else: #if move_y_from_center_and_decrease<0:
            y1 = 0
            y2 = int(init_image.shape[0] + half_diff_y*move_y_from_center_and_decrease)

        y1 = max(0, y1)
        y2 = min(init_image.shape[0], y2)

        half_diff_x = (y2-y1)/2
        center_x = init_image.shape[1]/2

        x1 = max(0, int(center_x-half_diff_x+half_diff_x*move_x_from_center))
        x2 = min(init_image.shape[1], int(x1+(y2-y1)))

    else:
        print('Rotate Image!! bbox is incorrect!')

    return x1, y1, x2, y2

def imcrop(img, bbox):
    if bbox is None:
        return img
    else:
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)

        return img[y1:y2, x1:x2]


def crop_and_resize_match_data(match_data, bbox=None, original_and_goal_dim=None):
    """
    Args:
        match_data: dict that contains data of feature point matches
                    (points as [x,y])
        bbox: bbox that determines cropped area of image data
        original_and_goal_dim: tuple of image sizes of (original, target) image
                               (shape as (y,x) rows and columns)
    Returns: Updated dict with match data. Points are also converted to numpy array!
    """
    pointsA = np.array(match_data['pointsA'])
    pointsB = np.array(match_data['pointsB'])

    # crop
    if bbox is not None:
        # filter point pair that lie outside bbox and move points to new coordinate system
        mask = []
        for pa, pb in zip(pointsA, pointsB):  # bbox = x1, y1, x2, y2 and points = x, y
            if (bbox[0] < pa[0] < bbox[2] and bbox[1] < pa[1] < bbox[3] and  # pointA inside of crop
               bbox[0] < pb[0] < bbox[2] and bbox[1] < pb[1] < bbox[3]):     # pointB inside of crop
                mask.append(True)
            else:
                mask.append(False)
        points_a_cropped = pointsA[mask] - bbox[0:2]
        points_b_cropped = pointsB[mask] - bbox[0:2]
    else:
        points_a_cropped = pointsA
        points_b_cropped = pointsB

    # calculate positions on resized image
    if original_and_goal_dim is not None:
        original_dim, goal_dim = original_and_goal_dim  # shapes: row, col (y,x)!
        points_a_cropped[:, 0] *= goal_dim[1] / original_dim[1] # points (x,y)
        points_a_cropped[:, 1] *= goal_dim[0] / original_dim[0]
        points_b_cropped[:, 0] *= goal_dim[1] / original_dim[1]
        points_b_cropped[:, 1] *= goal_dim[0] / original_dim[0]

    match_data['pointsA'] = points_a_cropped
    match_data['pointsB'] = points_b_cropped
    return match_data

def crop_and_resize_point_cloud(point_cloud, bbox=None, original_and_goal_dim=None):
    """
    Args:
        point_cloud: Array of points in (y,x) form (rows and columns)
        bbox: bounding box (x1, y1, x2, y2)
        original_and_goal_dim: shapes of (original, new) image in form (y,x)
    Returns:
    """
    point_cloud = point_cloud.cpu().numpy() if torch.is_tensor(point_cloud) else point_cloud
    if bbox is not None:
        # filter point pair that lie outside bbox and move points to new coordinate system
        mask = []  # Todo: not tested yet
        for point in point_cloud:  # bbox = x1, y1, x2, y2 and points= y,x (rows, columns)
            mask.append(bbox[0] < point[1] < bbox[2] and bbox[1] < point[0] < bbox[3])
        point_cloud_cropped = point_cloud[mask] - (bbox[1], bbox[0])
    else:
        point_cloud_cropped = point_cloud

    # calculate positions on resized image
    if original_and_goal_dim is not None:
        original_dim, goal_dim = original_and_goal_dim  # shapes: row, col (y,x)!
        point_cloud_cropped[:, 0] *= goal_dim[0] / original_dim[0]  # point cloud (y,x)
        point_cloud_cropped[:, 1] *= goal_dim[1] / original_dim[1]
    return point_cloud_cropped






def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
               (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2


def point_cloud_orientation_correction(torch_point_cloud, camera_synthetic_init):
    # torch_point_cloud_rotated = torch_point_cloud.clone() @ camera_synthetic_init.R
    return torch_point_cloud  #_rotated
#  Todo: Understand need for rotation and why different image sizes seem to require
#        different corrections. Implement adequate rot for other image sizes than 128 aswell or find
#        code/error that causes different behaviour for different image sizes


def get_sample_list(directory, glob_pattern='*.tiff'):
    '''
    Go through dir and return the samples encountered
    Args:
        directory: Dir in which to look for the files
        glob_pattern: Pattern of the files to be considered
    Returns:
        Sorted list of the stems of the files that match the pattern
    '''
    directory = Path(directory)
    file_list = list(directory.glob(glob_pattern))
    return sorted([file.stem for file in file_list])
