from typing import List, Tuple, Union

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import Sequence

from tools_generate.data import ds_to_list
from tools_generate.image import generate_outputs, get_graph_outputs
from tools_generate.PolyGraph import PolyGraph
import networkx as nx
from tools_generate.NodeContainer import NodeContainer
from .TestType import TestType


class DataGenerator(Sequence):
    """Generates training/validation data."""

    def __init__(self, config, ImageType=None, augmented: bool = False):

        if ImageType:
            self.grey_scale_process_folder = ImageType
        else:
            self.grey_scale_process_folder = config.grey_scale_process_folder

        self.num_data = config.num_train
        self.ds = config.training_ds
        # dimensions
        self.batch_size = config.batch_size
        self.img_dims_data_folder = config.img_dims_data_folder
        self.process_image_dim = config.process_image_dim
        self.input_channels = config.input_channels
        self.output_channels = config.output_channels

        # data_augmentation
        self.augmentation_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
        )
        self.augmented = augmented
        self.augmenter = (
            ImageDataGenerator(**self.augmentation_args) if augmented else None
        )

    def __len__(self):
        """Denotes the number of batches per epoch
        i.e. number of steps per epoch."""
        return int(np.floor(self.num_data / self.batch_size))

    def on_epoch_end(self):
        self.ds # ToDo: May need to fix the order here, for a sorted order! Is fixed! --> in get_batch_data ds.skip
        #self.ds = self.ds.shuffle(self.num_data, reshuffle_each_iteration=False)

    def __getitem__(
        self, batch_nr: int
    ) -> Union[
        np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ]:
        """
        Returns the i-th batch.
        :param i: batch index
        :return: X and Y when fitting. X only when predicting
        """
        rgb_image, grey_scale_image, graph_features = self.get_batch_data(batch_nr)

        return rgb_image, grey_scale_image, graph_features

    def get_batch_data(self, b: int):
        batch_fps = self.ds.skip(b * self.batch_size).take(self.batch_size)

        skel_fps = ds_to_list(batch_fps)  #to access to the corresponding folder
        graph_fps = [fp.replace("skeleton", "graphs").replace(".png", ".json") for fp in skel_fps]
        rgb_fps = [fp.replace("skeleton", "cropped") for fp in skel_fps]
        grey_fps = [fp.replace("skeleton", self.grey_scale_process_folder) for fp in skel_fps]

        rgb_image = self._get_image(rgb_fps, is_color=True)
        grey_scale_image = self._get_image(grey_fps, is_color=False)
        graph_features = self._generate_graph_related_features(graph_fps)
        #graph_features = self._generate_graph_related_feature_dictionary(graph_fps)

        return rgb_image, grey_scale_image, graph_features

    def _get_image(self, skel_fps: List[str], is_color = False) -> np.ndarray:
        """
        Generates normalised tensors of the skeletonised images.
        :param skel_fps: filepaths to the skeletonised images
        :return: skeletonised image tensor, normalised
        """
        if is_color:
            batch_color_image = np.empty((self.batch_size, self.process_image_dim, self.process_image_dim, 3), dtype=np.int)
            for i, fp in enumerate(skel_fps):
                color_image = img_to_array(load_img(fp, color_mode="rgb", target_size=[self.process_image_dim, self.process_image_dim], interpolation='nearest'), dtype=np.int)
                batch_color_image[i, :, :, :] = color_image
            return batch_color_image

        else:
            batch_grey_scale_image = np.empty((self.batch_size,  self.process_image_dim, self.process_image_dim))
            for i, fp in enumerate(skel_fps):
                grey_scale_image = img_to_array(load_img(fp, color_mode="grayscale", target_size=[self.process_image_dim,self.process_image_dim], interpolation='nearest'), dtype=np.float32)
                batch_grey_scale_image[i, :, :] = grey_scale_image.squeeze()
            return batch_grey_scale_image

    def _generate_graph_related_features(
        self, graph_fps: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates all Graph related features (node_pos, node_type, ... adj, ...), whereas the size depends on the actual loaded graph.
        :param graph_fps: filepaths to the graph objects
        :return: node position, node degree and node type tensors
        """
        batch_nx_graph = []
        batch_resized_node_pos = []

        batch_node_container = []

        batch_node_degrees = []
        batch_node_types = []
        batch_stacked_adj_matr = []


        for i, fp in enumerate(graph_fps):
            graph = PolyGraph.load(fp)
            output_matrices = get_graph_outputs(graph)

            node_pos = output_matrices["node_pos"]
            node_degrees = self._cap_degrees(output_matrices["degrees"])
            node_types = output_matrices["node_types"]
            stacked_adj_matr = output_matrices["stacked_adj_matrix"]

            # resize node positions:
            resized_node_pos = np.empty((node_pos.shape), dtype=np.uint8)
            resized_node_pos[:, 0] = np.round((node_pos[:, 0] / self.img_dims_data_folder) * self.process_image_dim, 0)
            resized_node_pos[:, 1] = np.round((node_pos[:, 1] / self.img_dims_data_folder) * self.process_image_dim, 0)
            # update node positions in graph
            pos_dictionary = {i: node_pos for i, node_pos in enumerate(resized_node_pos.astype(int))}
            nx.set_node_attributes(graph, pos_dictionary, 'pos')

            node_container = NodeContainer(graph=graph)
            batch_nx_graph.append(graph)
            batch_resized_node_pos.append(resized_node_pos.astype(int))
            batch_node_container.append(node_container)
            batch_node_degrees.append(node_degrees.astype(int))
            batch_node_types.append(node_types.astype(int))
            batch_stacked_adj_matr.append(stacked_adj_matr)

        return batch_nx_graph, batch_resized_node_pos, batch_node_degrees, batch_node_container, batch_stacked_adj_matr

    def _generate_graph_related_feature_dictionary(
        self, graph_fps: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ##ToDo: Not Used yet, since variables would be needed to be splited later for the whole batch, therefore _generate_graph_related_features is used
        """
        Generates all Graph related features (node_pos, node_type, ... adj, ...), whereas the size depends on the actual loaded graph.
        :param graph_fps: filepaths to the graph objects
        :return: node position, node degree and node type tensors
        """
        batch_graph_features = []

        for i, fp in enumerate(graph_fps):
            graph = PolyGraph.load(fp)
            output_matrices = get_graph_outputs(graph)

            node_pos = output_matrices["node_pos"]
            # node_degrees = self._cap_degrees(output_matrices["degrees"])
            # node_types = output_matrices["node_types"]
            # stacked_adj_matr = output_matrices["stacked_adj_matrix"]

            # resize node positions:
            resized_node_pos = np.empty((node_pos.shape), dtype=np.uint8)
            resized_node_pos[:, 0] = np.round((node_pos[:, 0] / self.img_dims_data_folder) * self.process_image_dim, 0)
            resized_node_pos[:, 1] = np.round((node_pos[:, 1] / self.img_dims_data_folder) * self.process_image_dim, 0)

            output_matrices["node_pos"] = resized_node_pos.astype(int)
            batch_graph_features.append(output_matrices)

        return batch_graph_features

    @staticmethod
    def _cap_degrees(degrees):
        """Cap values at 4."""
        cap_value = 4
        degrees[degrees > cap_value] = cap_value
        return degrees
