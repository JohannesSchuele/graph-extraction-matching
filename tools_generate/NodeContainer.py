import cv2
import numpy as np

#ToDo needs to go in config file
image_length = 128
mask_radius = 102.5/2 if image_length == 128 else 102.5
border_size = 2
border_radius =  int(mask_radius - border_size)
image_centre = (int(image_length / 2), int(image_length / 2))



from tools_generate.node_classifiers import NodeTypes
from tools_generate.PolyGraph import PolyGraph


def flip_node_coordinates(list_of_nodes_yx):
    return [[yx[1], yx[0]] for yx in list_of_nodes_yx]


def sort_list_of_nodes(unsorted):
    sorted_tuple = sorted(enumerate(unsorted), key=lambda x: [x[1][0], x[1][1]])
    indices, sorted_nodes = zip(*sorted_tuple)
    return indices, list(sorted_nodes)


def get_border_coordinates() -> np.ndarray:
    """
    The border zone is used to differentiate between valid and invalid nodes.
    """
    border_mask = np.zeros((image_length, image_length), np.float32)
    cv2.circle(
        border_mask, image_centre, int(border_radius), (1.0, 0.0, 0.0), border_size
    )

    return np.argwhere(border_mask).tolist()


border_coordinates = get_border_coordinates()


class NodeContainer(object):
    """
    Container object for all the nodes in a graph.
    Auto-classification of border nodes is carried out upon initialisation.
    """

    def __init__(
        self, crossing_nodes=None, end_nodes=None, border_nodes=None, graph: PolyGraph = None, graph_fp: str='' ):

        self._crossing_nodes_yx = crossing_nodes if crossing_nodes else []
        self._end_nodes_yx = end_nodes if end_nodes else []
        self._border_nodes_yx = border_nodes if border_nodes else []

        self._all_nodes_yx = []
        self._node_types = []

        if isinstance(graph, PolyGraph):
            self._load_from_graph(graph)
            self._classify_nodes()
        elif graph_fp:
            self._load_from_graph_fp(graph_fp)
            self._classify_nodes()
        else:
            self._check_border_nodes()
            self._concat_all_nodes()

        #self._sort_all_nodes()

    def _load_from_graph_fp(self, graph_fp):
        """
        Loads all nodes and their corresponding types from graph file path.
        """
        graph = PolyGraph.load(graph_fp)

        self.all_nodes_xy = graph.positions
        self.node_types = graph.node_types

    def _load_from_graph(self, graph: PolyGraph):
        """
        Loads all nodes and their corresponding types from graph.
        """
        self.all_nodes_xy = graph.positions
        self.node_types = graph.node_types

    def _classify_nodes(self):
        """
        Given a list of all nodes and the node types,
        classifies the nodes according to type and fills the corresponding
        node lists: border_nodes_yx, crossing_nodes_yx, end_nodes_yx
        """
        for node_type, [x, y] in self.data:
            if node_type == NodeTypes.BORDER:
                self._border_nodes_yx.append([y, x])
            elif node_type == NodeTypes.CROSSING:
                self._crossing_nodes_yx.append([y, x])
            elif node_type == NodeTypes.END:
                self._end_nodes_yx.append([y, x])

            #self._node_types.append(node_type) # ToDo: check

    def _check_border_nodes(self):
        """
        Reclassifies end nodes that lie on the border to border nodes.
        Note: does NOT update self.node_types
        """
        for _, yx in enumerate(self.end_nodes_yx):
            if yx in border_coordinates:
                self._end_nodes_yx.remove(yx)
                self._border_nodes_yx.append(yx)

    def _concat_all_nodes(self) -> None:
        """
        Sets the list of all nodes,
        as well as the corresponding node types
        """
        self._all_nodes_yx = (
            self.crossing_nodes_yx + self.end_nodes_yx + self.border_nodes_yx
        )
        self._node_types = (
            [NodeTypes.CROSSING] * self.num_crossing_nodes
            + [NodeTypes.END] * self.num_end_nodes
            + [NodeTypes.BORDER] * self.num_border_nodes
        )

    def _sort_all_nodes(self):
        """
        Sorts the lists all_nodes and node_types only.
        """
        unsorted_nodes_xy = self.all_nodes_xy
        unsorted_types = self.node_types

        indices, sorted_nodes = sort_list_of_nodes(unsorted_nodes_xy)

        self.all_nodes_xy = sorted_nodes
        self.node_types = [unsorted_types[i] for i in indices]

    def add_helper_node(self, helper_xy):
        x, y = helper_xy

        if helper_xy in border_coordinates:
            self._border_nodes_yx.append([y, x])
            self._node_types.append(NodeTypes.BORDER)
        else:
            self._crossing_nodes_yx.append([y, x])
            self._node_types.append(NodeTypes.CROSSING)

        self._all_nodes_yx.append([y, x])

        self._sort_all_nodes()

    # yx nodes
    @property
    def crossing_nodes_yx(self):
        return self._crossing_nodes_yx.copy()

    @property
    def end_nodes_yx(self):
        return self._end_nodes_yx.copy()

    @property
    def border_nodes_yx(self):
        return self._border_nodes_yx.copy()

    @property
    def all_nodes_yx(self):
        return self._all_nodes_yx.copy()

    # xy nodes
    @property
    def crossing_nodes_xy(self):
        return flip_node_coordinates(self._crossing_nodes_yx)

    @property
    def end_nodes_xy(self):
        return flip_node_coordinates(self._end_nodes_yx)

    @property
    def border_nodes_xy(self):
        return flip_node_coordinates(self._border_nodes_yx)

    @property
    def all_nodes_xy(self):
        return flip_node_coordinates(self._all_nodes_yx)

    @all_nodes_xy.setter
    def all_nodes_xy(self, nodes):
        self._all_nodes_yx = flip_node_coordinates(nodes)

    # number of nodes
    @property
    def num_crossing_nodes(self):
        return len(self.crossing_nodes_yx)

    @property
    def num_end_nodes(self):
        return len(self.end_nodes_yx)

    @property
    def num_border_nodes(self):
        return len(self.border_nodes_yx)

    @property
    def num_all_nodes(self):
        return len(self.all_nodes_yx)

    # other graph attributes
    @property
    def node_types(self):
        return [x.value for x in self._node_types]

    @node_types.setter
    def node_types(self, values):
        if 0 in values:
            corrected_vals = [3 if v == 0 else v for v in values]
            values = corrected_vals

        self._node_types = [NodeTypes(v) for v in values]

    @property
    def data(self):
        """
        Returns a list of tuple: (node type, node xy coordinates).
        """
        return zip(self._node_types, self.all_nodes_xy)