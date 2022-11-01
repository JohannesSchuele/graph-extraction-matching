import torch
import torch.nn as nn
# rendering components
from tools_graph.utilz_analysis import plot_graph_on_img
import copy
from models.image import Image
from tools_graph.match_orb_tools import BRIEF, match
from skimage.transform import AffineTransform, EssentialMatrixTransform
from skimage.measure import ransac
from tools_graph.plot_functs import plot_graph_matches_color_inliers_outliers
from tools_generate.image_processing import skeletonise_and_clean
from configs.plot.config_plots import *
from configs.config import *


class UpComingDataGenerator(nn.Module):

    def __init__(self, data_generator, device, task=None):
        super().__init__()

        self.device = device
        self.data_generator = data_generator
        self.point_cloud_init = None
        self.world_point_cloud_map = None
        self.current_batch_object = None
        self.currently_loaded_batch_nr = None  # do not touch that variable!! It tracks the currently loaded batch in order to avoid reloading
        self.currently_loaded_batch_item = None  # do not touch that variable!! It tracks the currently loaded batch in order to avoid reloading
        self.desired_item_in_batch_ofcurrentinterest = 0
        self.desired_batch_nr_ofcurrentinterest = 0
        self.nr_of_call = 0
        self.image_size = data_generator.process_image_dim
        self.task = {'UpComingDataGenerator plot matches between images - for final sample': False,
                     'UpComingDataGenerator matches and make Gif - for all upcoming samples': False,
                     'UpComingDataGenerator number of minimal matches': 65,
                     'UpComingDataGenerator reskeletonize grey image': True,
                      'UpComingDataGenerator plot reskeletonizd grey image': False
                     }
        if task is not None:
            self.task.update(task)

    def next_image_of_interest(self, current_graph_image,
                               current_image_rgb, current_image_grey, nr_of_call=0, task_update = None):
        if task_update is not None:
            self.task.update(task_update)
        min_matches = self.task['UpComingDataGenerator number of minimal matches']
        sufficient_matches = True
        matches_prev = None
        itr = 0

        if self.task['UpComingDataGenerator matches and make Gif - for all upcoming samples']:
            GifMatches = GifMaker(
                name_of_gif='matches_on_image_with_outliers_' + str(nr_of_call) + '_ith_call')

        while sufficient_matches:
            itr = +1
            prev_batch = self.currently_loaded_batch_nr
            prev_item = self.currently_loaded_batch_item
            desired_item_in_batch = self.currently_loaded_batch_item + 1
            next_image_rgb, next_image_grey, next_graph, \
            next_pos, next_node_degrees, next_node_container, \
            next_adj_matrix, next_stacked_adj_features = self.load_root_image_data(
                desired_batch_nr=prev_batch,
                item_in_batch=desired_item_in_batch)
            d1 = BRIEF(current_image_grey, copy.deepcopy(current_graph_image.node_pos), orientations=None,
                       n=512, patch_size=25, sigma=0.1, mode='uniform',
                       sample_seed=42)
            d2 = BRIEF(next_image_grey, copy.deepcopy(next_pos), orientations=None, n=512, patch_size=25, sigma=0.1,
                       mode='uniform',
                       sample_seed=42)
            matches_brute_force = match(d1, d2, cross_check=True, max_distance=np.inf)
            inliers = self.check_inliers_ransac(matches_brute_force, current_graph_image.node_pos, next_pos)
            matches = matches_brute_force[inliers]

            if self.task['UpComingDataGenerator matches and make Gif - for all upcoming samples']:
                fig_matches_on_image_rgb = plot_graph_matches_color_inliers_outliers(current_image_rgb, next_image_rgb,
                                                                                     copy.deepcopy(
                                                                                         current_graph_image.node_pos),
                                                                                     current_graph_image.adj,
                                                                                     copy.deepcopy(next_pos),
                                                                                     next_adj_matrix,
                                                                                     matches_brute_force=matches_brute_force,
                                                                                     inliers=inliers, color_edge=None,
                                                                                     title='matches with outliers' + str(
                                                                                         nr_of_call) + '_ith_call_' + str(
                                                                                         itr) + '_ith_itr')
                save_figure(fig=fig_matches_on_image_rgb, name_of_figure='matches_with_images_and_outliers' +
                                                                         str(nr_of_call) + '_ith_call_' + str(itr) + '_ith_itr')
                GifMatches.add_figure(fig=fig_matches_on_image_rgb)
                # fig_matches_on_image_grey = plot_graph_matches_color_inliers_outliers(self.current_image_grey, next_image_grey,
                #                                 copy.deepcopy(self.current_graph_image.node_pos),
                #                                 self.current_graph_image.adj, copy.deepcopy(next_pos),
                #                                 next_adj_matrix,
                #                                 matches_brute_force=matches_brute_force,
                #                                 inliers=inliers,
                #                                 color_edge=None)

            sufficient_matches = matches.shape[0] > min_matches
            if sufficient_matches:
                matches_prev = matches
            else:  # ToDo: update the image uptake from the previous time step
                if matches_prev is not None:
                    # we try to move forward by decreasing the matches
                    # as long we can guarantee a minimum match dimension, all is fine
                    matches = matches_prev
                    self.desired_item_in_batch_ofcurrentinterest = prev_item
                    self.desired_batch_nr_ofcurrentinterest = prev_batch
                else:
                    # here we can't guarantee the minimum match dimension,
                    # however we still prefer to move forward to the next image!
                    matches = matches
                    self.desired_item_in_batch_ofcurrentinterest = desired_item_in_batch
                    self.desired_batch_nr_ofcurrentinterest = prev_batch
                print('batch nr  in search:', self.desired_batch_nr_ofcurrentinterest, 'and item nr : ',
                      self.desired_item_in_batch_ofcurrentinterest)
        if self.task['UpComingDataGenerator plot matches between images - for final sample']:
            fig_matches_on_image_rgb = plot_graph_matches_color_inliers_outliers(current_image_rgb, next_image_rgb, copy.deepcopy(current_graph_image.node_pos),
                                                                             current_graph_image.adj, copy.deepcopy(next_pos), next_adj_matrix, matches_brute_force=matches_brute_force,
                                                                             inliers=inliers, color_edge=None, title='matches with outliers' + str(
                                                                                 nr_of_call) + '_ith_call_' + str(itr) + '_ith_itr')
            print('Number of matches :', matches.shape[0])
            save_figure(fig=fig_matches_on_image_rgb, name_of_figure='matches_with_images_and_outliers' +
                                                                 str(nr_of_call) + '_ith_call_' + str(itr) + '_ith_itr')
        if self.task['UpComingDataGenerator matches and make Gif - for all upcoming samples']: GifMatches.close_writer()
        return matches, self.desired_batch_nr_ofcurrentinterest, self.desired_item_in_batch_ofcurrentinterest

    def check_inliers_ransac(self, matches, pos1, pos2):
        src = np.array(pos1[matches[:, 0]])
        dst = np.array(pos2[matches[:, 1]])
        model = AffineTransform()
        # model.estimate(src, dst)
        model_essential = EssentialMatrixTransform()
        model_essential.estimate(src, dst)
        # model = model_essential
        # robustly estimate affine transform model with RANSAC
        model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                                       residual_threshold=4, max_trials=50)
        outliers = inliers == False
        return inliers

    def update_point_cloud(self):
        self.point_cloud_world_init.update_padded(new_points_padded=None, new_normals_padded=None,
                                                  new_features_paddeed=None)  # ToDo implement pontcloud update with merging the current pointcloud to the world point cloud

    def load_root_image_data(self, desired_batch_nr, item_in_batch):
        if not (self.currently_loaded_batch_nr == desired_batch_nr):
            self.current_batch_object = self.data_generator.__getitem__(batch_nr=desired_batch_nr)
            # current_batch_object with the structure: image_rgb, image_grey, graph
            self.currently_loaded_batch_nr = desired_batch_nr  # to track which batch nr is currently loaded
        if item_in_batch >= self.data_generator.batch_size:
            desired_batch_nr = desired_batch_nr + 1
            item_in_batch = 0
            self.current_batch_object = self.data_generator.__getitem__(batch_nr=desired_batch_nr)
            self.currently_loaded_batch_nr = desired_batch_nr  # to track which batch nr is currently loaded
        self.currently_loaded_batch_item = item_in_batch
        image_rgb = self.current_batch_object[0][item_in_batch]
        image_grey = self.current_batch_object[1][item_in_batch]
        batch_graph = self.current_batch_object[2]
        batch_nx_graph = batch_graph[0]
        batch_pos = batch_graph[1]
        batch_node_degrees = batch_graph[2]
        batch_node_container = batch_graph[3]
        batch_stacked_adj_matr = batch_graph[4]
        # graph element is build up in a list by the following structure ## [batch_resized_node_pos, batch_node_degrees, batch_node_types, batch_stacked_adj_matr]
        nx_graph = batch_nx_graph[item_in_batch]
        pos = batch_pos[item_in_batch]
        node_degrees = batch_node_degrees[item_in_batch]
        node_container = batch_node_container[item_in_batch]
        stacked_adj_matr = batch_stacked_adj_matr[item_in_batch]
        adj_matrix = stacked_adj_matr[0]
        stacked_adj_features = stacked_adj_matr[1:]
        if PLOT_FIRST_IMAGES:
            fig_loaded_image = plot_graph_on_img(image_rgb, pos, adj_matrix)
        return image_rgb, image_grey, nx_graph, pos, node_degrees, node_container, adj_matrix, stacked_adj_features

    def load_and_create_graph_image_object(self, desired_batch_nr, item_in_batch, r_rotation=None, t_translation=None):
        image_rgb, image_grey, nx_graph, pos, node_degrees, node_container, adj_matrix, stacked_adj_features = self.load_root_image_data(
            desired_batch_nr, item_in_batch)
        brief_features = self.get_orb_features_of_image(image_grey,
                                                        pos)  # ToDo: include node_degrees etc. in Image Object and actual orb features
        # ToDo save nx_graoh in Image!!
        graph_image = Image(node_pos=pos, nx_graph=nx_graph, node_container=node_container, adjaceny=adj_matrix,
                            r_rotation=r_rotation, t_translation=t_translation,
                            batch_nr=self.currently_loaded_batch_nr, item_in_batch=self.currently_loaded_batch_item,
                            device=self.device,
                            image_size=self.image_size,
                            attributes_orb=brief_features)
        if self.task['UpComingDataGenerator reskeletonize grey image']:
            skeletonized_image_grey = skeletonise_and_clean(
                thr_image =image_grey, plot=self.task['UpComingDataGenerator plot reskeletonizd grey image'], save =False, directory='')
            image_grey = skeletonized_image_grey

        return graph_image, image_rgb, image_grey

    def get_orb_features_of_image(self, image_grey, pos):
        brief_features = BRIEF(image_grey, copy.deepcopy(pos), orientations=None, n=512, patch_size=25, sigma=0.1,
                               mode='uniform',
                               sample_seed=42)
        return brief_features

    def new_image(self, current_gim):
        self.total_nr_of_observed_gims += 1
        self.keys.append(self.total_nr_of_observed_gims)
        self.dictonary_of_graphs.update({self.total_nr_of_observed_gims: current_gim})

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


    def close_gifs(self):
        if PLOT_WORLD_MAP:
            self.GifWorldMApView.close_writer()
            self.GifWorldMAp2D.close_writer()
        else:
            print('If GIFS are supposed to be created, the variable PLOT_WORLD_MAP in config_plots.py has to be set!')




