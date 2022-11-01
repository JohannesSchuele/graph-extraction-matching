import networkx as nx
import numpy as np
import torch
import copy
from pytorch3d.structures import Pointclouds
from models.fragments import FragmentMap
from tools_generate.node_classifiers import NodeTypes
from tools_generate.PolyGraph import PolyGraph

class ORBFeatures():
    def __init__(self, saving_dimension, first_features = None, orb_feature_matrix_and_hash=None, device='cpu'):

        self.saving_dimension = saving_dimension
        self.device = device
        if first_features is not None and orb_feature_matrix_and_hash  is None:
            self.feature_matrix = torch.zeros((first_features.shape[0], first_features.shape[1], self.saving_dimension), device= self.device, dtype=bool)
            # hash_index_map always indicates the indices where the matirx starts to be unfilled, or replaced!
            # if data needs to be accessed, hash_index_map-1 gives the latest entry in the matrix
            self.hash_index_map = torch.zeros((first_features.shape[0], 2), device=self.device, dtype=int)
            self.hash_index_map[:, 0] = torch.arange(0, first_features.shape[0], device=self.device, dtype=int)
            self.add_new_features_to_existing_points(first_features)

        if first_features is None and orb_feature_matrix_and_hash  is not None:
            self.feature_matrix = orb_feature_matrix_and_hash[0]
            self.hash_index_map = orb_feature_matrix_and_hash[1]

    def add_new_features_to_existing_points(self, new_features, matches=None):
        self.hash_index_map % self.saving_dimension  #To ensure that saving_dimension is not exceeded
        if matches is not None:
            if matches.shape[1] == 2:
                self.feature_matrix[self.hash_index_map[matches[:, 0], 0], :, self.hash_index_map[matches[:, 0], 1]] = new_features[matches[:, 1]].to(device=self.device)
                self.hash_index_map[matches[:, 0], 1] = self.hash_index_map[matches[:, 0], 1] + 1
            else:
                self.feature_matrix[self.hash_index_map[matches[:, 0], 0], :, self.hash_index_map[matches[:, 0], 1]] = new_features.to(device=self.device)
                self.hash_index_map[matches[:, 0], 1] = self.hash_index_map[matches[:, 0], 1] + 1
        else:
            self.feature_matrix[self.hash_index_map[:, 0], :, self.hash_index_map[:, 1]] = new_features.to(device=self.device)
            self.hash_index_map[:, 1] = self.hash_index_map[:, 1]+1

    def expand_for_new_points(self, first_features_new_points):
        self.feature_matrix = torch.vstack((self.feature_matrix, torch.zeros((first_features_new_points.shape[0], first_features_new_points.shape[1], self.saving_dimension), device=self.device, dtype=bool)))
        indices = torch.arange(self.hash_index_map.shape[0], self.hash_index_map.shape[0]+first_features_new_points.shape[0], device= self.device, dtype= int)
        hash_index_tmp = torch.zeros((first_features_new_points.shape[0], 2), device=self.device, dtype=int)
        self.hash_index_map = torch.vstack((self.hash_index_map, hash_index_tmp))
        self.hash_index_map[:, 0] = torch.arange(0, self.hash_index_map.shape[0], device=self.device)
        self.add_new_features_to_existing_points(first_features_new_points, matches= indices)

    def expand_for_feature_class(self, new_feature_class):
        new_feature_matrix = new_feature_class[0]
        new_feature_matrix_hash = new_feature_class[1]
        assert new_feature_matrix.shape[2] is self.saving_dimension
        self.feature_matrix = torch.vstack((self.feature_matrix, new_feature_matrix))
        self.hash_index_map = torch.vstack((self.hash_index_map, new_feature_matrix_hash))
        self.hash_index_map[:, 0] = torch.arange(0, self.hash_index_map.shape[0], device=self.device)


    def get_sub_class(self, indices):
        hash_index_map = self.hash_index_map[indices]
        hash_index_map[:, 0] = torch.arange(0, hash_index_map.shape[0], device=self.device)
        return self.__class__(self.saving_dimension, orb_feature_matrix_and_hash= (self.feature_matrix[indices], hash_index_map), device=self.device)

    def get_all_features_with_hash(self):
        return (self.feature_matrix, self.hash_index_map)


class   PointCloudMap(Pointclouds):
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
    def __init__(self, PointCloud=None, points=None, normals=None, features=None,
                 types: torch.Tensor=None,
                 remove_node_type=None,
                 fragments_map=None, orb_feature_class=None, new_orb_features=None,
                 nx_graph:PolyGraph =None, size_points_prev=None, device='cpu') -> None:
        """
        Args:
            fragments_map is setted up to be optional, and is also retrievable later and assigned!
        """
        self.device = device
        self.types = types
        if orb_feature_class is None and new_orb_features is not None:
            self.orb_feature_class = ORBFeatures(saving_dimension=15, first_features=new_orb_features, device=self.device)

        elif isinstance(orb_feature_class, ORBFeatures):
            self.orb_feature_class = orb_feature_class
        else: self.orb_feature_class = None

        #self.orb_feature_class = orb_feature_class
        # else:
        #     self.orb_feature_class = orb_feature_class
        if not(PointCloud):
            super().__init__(points=points, normals=normals, features=features)
        else:
            super().__init__(points=PointCloud.points_list(), normals=PointCloud.normals_list(), features=PointCloud.features_list())
        if device == 'cpu':
            self.cpu()
        else:
            self.cuda()

        if nx_graph is True:
            self.nx_graph = PolyGraph()
            self.nx_graph.add_nodes_from(range(0, self.points_packed().shape[0]))
            #self.nx_graph.add_nodes_from(torch.arange(0, self.points_packed().shape[0], dtype=int))

        #elif nx_graph is nx.Graph():
        elif isinstance(nx_graph, PolyGraph): #if it is an instance of PolyGraph it is also an instance of nx.Graph
            # -> so if instance is checked to be a nx.Graph we ensure it is initialized as an PolyGraph
            self.nx_graph = nx_graph
            nx.set_edge_attributes(self.nx_graph, 0, "ith_observations")
        else:
            self.nx_graph = None

        if size_points_prev is not None:
            self.size_points_before_update = size_points_prev
        else:
            self.size_points_before_update = copy.deepcopy(self.points_packed().shape[0])

        """
        Args: fragment_map is setted up to be optional, and is also retrievable later and assigned!
        """
        if isinstance(fragments_map, FragmentMap):
            self.fragments_map = fragments_map
            # We check if the tensors contain the same number of corresponding points!
            assert fragments_map.shape[2] == self.points_packed().shape[0]
        else:
            self.fragments_map = None

        if remove_node_type is not None:
            self._remove_nodes_of_type(remove_node_type=remove_node_type)


    def add_new_features_to_existing_points(self, new_features, matches):
        assert isinstance(self.orb_feature_class, ORBFeatures)
        self.orb_feature_class.add_new_features_to_existing_points(new_features, matches=matches)

    def update_fragments(self, indices, new_fragments_input = None, new_pix_to_face=None,
                          new_zbuf=None, new_bary_coords=None, new_dists=None):

        self.fragments_map.update_seen_fragments(indices, new_fragments_input=new_fragments_input, new_pix_to_face=new_pix_to_face,
                          new_zbuf=new_zbuf, new_bary_coords=new_bary_coords, new_dists=new_dists)

    def update_node_types(self, indices, new_types):
        self.types[indices] = new_types


    def update_point_cloud_based_on_mesh(self, mesh):
        """
        fragments_map needs to exist! based on that and the mesh, the new point positions and normals are calculated!
        """
        assert isinstance(self.fragments_map, FragmentMap)
        updated_point_cloud = self.fragments_map.update_point_cloud(updated_mesh=mesh)
        self.__init__(points=updated_point_cloud.points_list(), normals=updated_point_cloud.normals_list(),
                      features=updated_point_cloud.features_list(),
                      types=self.types,
                      fragments_map=self.fragments_map,
                      orb_feature_class=self.orb_feature_class, nx_graph=self.nx_graph,
                      size_points_prev=self.size_points_before_update,  device=self.device)
        return self


    def get_sub_points(self, indices):
        points = self.points_packed()[indices]
        normals = self.normals_packed()[indices]
        features = self.features_packed()[indices]
        types = self.types[indices]
        points = self._adjust_tensor_dimension(points)
        normals = self._adjust_tensor_dimension(normals)
        features = self._adjust_tensor_dimension(features)

        if isinstance(self.orb_feature_class, ORBFeatures):
            orb_feature_sub_class = self.orb_feature_class.get_sub_class(indices)
        else:
            orb_feature_sub_class = None

        if isinstance(self.nx_graph, PolyGraph):
            #https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.subgraph.html
            if indices[0] == True or indices[0] == False:
                prev_node_keys = np.arange(0, indices.shape[0])
                prev_node_keys = prev_node_keys[indices.clone().cpu().detach().numpy()]
                sub_nx_graph = self.nx_graph.subgraph(prev_node_keys).copy()
                # rename node keys of sub graph
                new_sub_node_keys = np.arange(0, prev_node_keys.shape[0])
                rename_mapping = dict(zip(prev_node_keys, new_sub_node_keys))
                new_nx_graph = nx.relabel_nodes(sub_nx_graph, rename_mapping)

        else:
            new_nx_graph = None

        if isinstance(self.fragments_map, FragmentMap):
            fragments_map_sub = self.fragments_map.get_sub_points(indices)
        else:
            fragments_map_sub = None

        return points, normals, features, types, fragments_map_sub, orb_feature_sub_class, new_nx_graph

    def get_sub_point_cloud(self, indices):
        points, normals, features, types, fragments_map_sub, orb_feature_sub_class, nx_graph = self.get_sub_points(indices)
        size_points_before_update = copy.deepcopy(points.shape[1])
        return self.__class__(points=points, normals=normals, features=features,
                              types = types,
                              fragments_map=fragments_map_sub,
                              orb_feature_class=orb_feature_sub_class, nx_graph=nx_graph,
                              size_points_prev=size_points_before_update, device=self.device)

    def keep_sub_points_in_map(self, indices):
        points, normals, features, types, fragments_map_sub, orb_features, nx_graph =self.get_sub_points(indices)
        size_points_before_update = copy.deepcopy(points.shape[1])
        self.__init__(points=points, normals=normals, features=features,
                      types= types,
                      fragments_map=fragments_map_sub,
                      orb_feature_class=orb_features, nx_graph=nx_graph,
                      size_points_prev=size_points_before_update,  device=self.device)

    def _remove_nodes_of_type(self, remove_node_type: NodeTypes=None, keep_node_type: NodeTypes=None):

        if remove_node_type is NodeTypes.BORDER:
            self.keep_sub_points_in_map(indices=torch.logical_or(self.types[:, 0], self.types[:, 1]))

        if remove_node_type is NodeTypes.END:
            self.keep_sub_points_in_map(indices=torch.logical_or(self.types[:, 0], self.types[:, 2]))

        if keep_node_type:
            self.keep_sub_points_in_map(indices=self.types[:, keep_node_type.value])


    def add_points(self, PointCloudMap=None, new_points=None, new_normals=None, new_features=None,
                   new_types=None, remove_node_type=None,
                   new_fragments_map=None, new_orb_features= None) -> "PointCloudMap":

        #self._remove_nodes_of_type(remove_node_type=NodeTypes.BORDER)
        # Is already done before matching!
        size_points_before_update = copy.deepcopy(self.points_packed().shape[0])

        if PointCloudMap:
            points = torch.cat((self.points_packed(), PointCloudMap.points_packed()))
            normals = torch.cat((self.normals_packed(), PointCloudMap.normals_packed()))
            features = torch.cat((self.features_packed(), PointCloudMap.features_packed()))
            types = torch.cat((self.types, PointCloudMap.types))
            points = self._adjust_tensor_dimension(points)
            normals = self._adjust_tensor_dimension(normals)
            features = self._adjust_tensor_dimension(features)
            self.orb_feature_class.expand_for_feature_class(PointCloudMap.get_orb_feature_matrix)
            if isinstance(self.fragments_map, FragmentMap):
                self.fragments_map.add_new_fragments(new_fragments_input=PointCloudMap.fragments_map)

            if isinstance(self.nx_graph, PolyGraph):
                # even the graph is extended with new nodes for all given points
                # the corresponding node types are checked and truncated in the remove_nod_type function
                self.nx_graph.add_nodes_from(range(size_points_before_update,
                                                   size_points_before_update + PointCloudMap.points_packed().shape[0]))
            self.__init__(points=points, normals=normals, features=features,
                          types=types,
                          remove_node_type= remove_node_type,
                          fragments_map= self.fragments_map,
                          orb_feature_class=self.orb_feature_class,
                          nx_graph=self.nx_graph, size_points_prev=self.size_points_before_update,
                          device=self.device)
             #ToDo: check if this also simply works
        else:
            points =self.points_list().append(new_points)
            types = torch.cat((self.types,new_types))
            if new_normals:
                normals = self.normals_list().append(new_normals)
            else:
                normals = self.normals_list().append(torch.full(new_points.shape, None))
            if new_features:
                features = self.features_list().append(new_features)
            else:
                features = self.features_list().append(torch.full(new_points.shape, None))
            if torch.is_tensor(new_orb_features):
                self.orb_feature_class.expand_for_new_points(new_orb_features)
            else:
                print('Error: Feature vector for all new nodes are needed!!')

            if isinstance(self.fragments_map, FragmentMap):
                self.fragments_map.add_new_fragments(new_fragments_input=new_fragments_map)

            if isinstance(self.nx_graph, PolyGraph):
                self.nx_graph.add_nodes_from(range(self.points_packed().shape[0], self.points_packed().shape[0]+PointCloudMap.points_packed().shape[0]))
            self.__init__(points=points, normals=normals, features=features,
                          types = types,
                          remove_node_type=remove_node_type,
                          fragments_map=self.fragments_map,
                          orb_feature_class=self.orb_feature_class, nx_graph=self.nx_graph, size_points_prev=self.size_points_before_update, device=self.device)

            self.size_points_before_update = size_points_before_update


            #ToDo add own feature vecotr for old feature points
     # def update_features(self, features):
     #     features_parsed = self._parse_auxiliary_input(features)
     #     self._features_list, super()._features_padded, features_C = features_parsed
     #     if features_C is not None:
     #         self._C = features_C

    def update_graph(self, new_graph: PolyGraph, matches_to_world_point_could, indices_of_new_nodes=None, size_of_new_points=None):

        if size_of_new_points is not None:
            indices = np.vstack((np.arange(0, size_of_new_points),
                                 np.arange(0, size_of_new_points))).astype(int).transpose()
            indices = torch.from_numpy(indices)
            indices[:, 0] = self.size_points_before_update+indices[:, 0]
            matches = list(matches_to_world_point_could)+list(indices)
            #matches = list(matches_to_world_point_could)
        elif indices_of_new_nodes is not None:
            print('Number of new point matches that are considered', indices_of_new_nodes.shape[0])
            indices_of_new_nodes[:, 0] = self.size_points_before_update+indices_of_new_nodes[:, 0]
            matches= list(matches_to_world_point_could)+list(indices_of_new_nodes)
        else:
            matches = list(matches_to_world_point_could)
        allocation_matches = [False for _ in range(new_graph.number_of_nodes())]
        for match in matches:
            allocation_matches[match[1]]= int(match[0])
        #ToDo add the new points also to the match!

        for edge in list(new_graph.edges(data=True)):
            node_1 = edge[0]
            node_2 = edge[1]
            if allocation_matches[node_1] and allocation_matches[node_2]:
                if (not self.nx_graph.has_node(allocation_matches[node_1])) or (not self.nx_graph.has_node(allocation_matches[node_2])):
                    print('We have a problem!!!', self.nx_graph.has_node(allocation_matches[node_1]), 'node :', allocation_matches[node_1])
                    print('We have a problem!!!', self.nx_graph.has_node(allocation_matches[node_2]), 'node :', allocation_matches[node_2])

                if self.nx_graph.has_edge(allocation_matches[node_1], allocation_matches[node_2]):
                    #ToDo add ith_observations
                    self.nx_graph[allocation_matches[node_1]][allocation_matches[node_2]]["ith_observations"] += 1
                else:
                    self.nx_graph.add_edge(allocation_matches[node_1], allocation_matches[node_2], ith_observations=1)
        #ToDo capsulate the following it an own function
        #

    @ property
    def get_nx_graph(self):
        return self.nx_graph

    @ property
    def get_orb_feature_matrix(self):
        return self.orb_feature_class.get_all_features_with_hash()


    @ property
    def get_node_types(self):
        types = torch.tensor([1, 2, 3])
        return torch.tensor([types[self.types[i]] for i in range(self.types.shape[0])])


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