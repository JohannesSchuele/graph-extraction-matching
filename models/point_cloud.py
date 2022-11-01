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
import networkx as nx
import numpy as np
from pytorch3d.structures import Pointclouds
#ToDo implement feature tracking functions to update and recieve features of intrest!!

class   PointCloudMap(Pointclouds):
    ##ToDO to delte!! is deprecated!!
    """
    Args:
        points:
            Can be either

            - List where each element is a tensor of shape (num_points, 3)
              containing the (x, y, z) coordinates of each point.
            - Padded float tensor with shape (num_clouds, num_points, 3).
        normals:
            Can be either

            - List where each element is a tensor of shape (num_points, 3)
              containing the normal vector for each point.
            - Padded float tensor of shape (num_clouds, num_points, 3).
        features:
            Can be either

            - List where each element is a tensor of shape (num_points, C)
              containing the features for the points in the cloud.
            - Padded float tensor of shape (num_clouds, num_points, C).
            where C is the number of channels in the features.
            For example 3 for RGB color.

    Refer to comments above for descriptions of List and Padded
    representations.
    """
    def __init__(self, PointCloud=None, points=None, normals=None, features=None, orb_features =None, nx_graph =True, device = None) -> None:
        if torch.is_tensor(orb_features):
            self.orb_features = orb_features
        else:
            self.orb_features = None

        if not(PointCloud):
            super().__init__(points=points, normals=normals, features=features)

        else:
            super().__init__(points=PointCloud.points_list(), normals=PointCloud.normals_list(), features=PointCloud.features_list())


        if nx_graph is True:
            self.nx_graph = nx.Graph()
            self.nx_graph.add_nodes_from(range(0, self.points_packed().shape[0].cpu().detach().numpy()))

        elif nx_graph is nx.Graph():
            self.nx_graph = nx_graph
        else:
            self.nx_graph = None

        if device == 'cpu':
            self.cpu()
        else:
            self.cuda()


    def get_sub_points(self, indices):
        points = self.points_packed()[indices]
        normals = self.normals_packed()[indices]
        features = self.features_packed()[indices]
        points = self._adjust_tensor_dimension(points)
        normals = self._adjust_tensor_dimension(normals)
        features = self._adjust_tensor_dimension(features)


        if torch.is_tensor(self.orb_features) and self.nx_graph is nx.Graph():
            orb_features = self.orb_features[indices]
            nx_graph = self.nx_graph.remove_nodes_from(indices)
            return points, normals, features, orb_features, nx_graph
        elif torch.is_tensor(self.orb_features) and self.nx_graph is None:
            orb_features = self.orb_features[indices]
            return points, normals, features, orb_features
        else:
            return points, normals, features, None
        #return points[None, :], normals[None, :], features[None,:]  # Dummy dimension is added as the default batch dimension



    def get_sub_point_cloud(self, indices):
        points, normals, features, orb_features, nx_graph = self.get_sub_points(indices)

        #return self.__class__(points=points[None, :], normals=normals[None, :], features=features[None, :], orb_features=orb_features)
        return self.__class__(points=points, normals=normals, features=features,
                              orb_features=orb_features, nx_graph=nx_graph)
        #return PointCloudMap(points=points[None, :], normals=normals[None, :], features=features[None, :])

    def keep_sub_points_in_map(self, indices):
        points, normals, features, orb_features, nx_graph =self.get_sub_points(indices)
        self.__init__(points=points, normals=normals, features=features, orb_features=orb_features, nx_graph=nx_graph)


    def add_points(self, PointCloudMap=None, new_points=None, new_normals=None, new_features=None, new_orb_features= None) -> "PointCloudMap":

        if PointCloudMap:
            points = torch.cat((self.points_packed(),PointCloudMap.points_packed()))
            normals = torch.cat((self.normals_packed(),PointCloudMap.normals_packed()))
            features = torch.cat((self.features_packed(), PointCloudMap.features_packed()))
            points = self._adjust_tensor_dimension(points)
            normals = self._adjust_tensor_dimension(normals)
            features = self._adjust_tensor_dimension(features)

            new_orb_features = torch.cat((self.orb_features, PointCloudMap.get_orb_features))

            self.__init__(points=points, normals=normals, features=features, orb_features= new_orb_features)

        else:
            points =self.points_list().append(new_points)
            if new_normals:
                normals = self.normals_list().append(new_normals)
            else:
                normals = self.normals_list().append(torch.full(new_points.shape, None))
            if new_features:
                features = self.features_list().append(new_features)
            else:
                features = self.features_list().append(torch.full(new_points.shape, None))
            if torch.is_tensor(new_orb_features):
                self.update_orb_features(new_features)
            else:
                print('Error: Feature vector for all new nodes are needed!!')

            self.__init__(points=points, normals=normals, features=features)


    #ToDo add own feature vecotr for old feature points
     # def update_features(self, features):
     #     features_parsed = self._parse_auxiliary_input(features)
     #     self._features_list, super()._features_padded, features_C = features_parsed
     #     if features_C is not None:
     #         self._C = features_C

    def update_graph(self, new_graph, matches_to_world_point_could, new_points):
        allocation_matches = [[False] for _ in range(new_graph.number_of_nodes())]
        for match in list(matches_to_world_point_could):
            allocation_matches[match[1]]= match[0]
        #ToDo add the new points also to the match!

        for edge in list(new_graph.edges_iter(data='weight', default=1)):
            node_1 = edge[0]
            node_2 = edge[1]
            if allocation_matches[node_1] and allocation_matches[node_1]:
                if self.nx_graph.has_edge(allocation_matches[node_1], allocation_matches[node_2]):
                    self.nx_graph.edges[allocation_matches[node_1], allocation_matches[node_1]]["ith_observations"]+=1
                else:
                    self.nx_graph.add_edge(allocation_matches[node_1], allocation_matches[node_2], ith_observations=1)





    def update_orb_features(self, new_orb_features):
        self.orb_features = torch.cat((self.orb_features, new_orb_features), dim=0)

    def set_orb_features(self, orb_features):
        if torch.is_tensor(orb_features):
            self.orb_features = orb_features

    @ property
    def get_orb_features(self):
        return self.orb_features

    @ property
    def get_nx_graph(self):
        return self.nx_graph

    @staticmethod
    def _adjust_tensor_dimension(tensor):
        if tensor.shape.__len__() == 1:
            tensor = tensor[None, None, :]
            return tensor
        elif tensor.shape.__len__() == 2:
            tensor =tensor[None, :]
            return tensor
        elif tensor.shape.__len__() == 3:
            return tensor
        else:
            print('Dimension Error!!!')