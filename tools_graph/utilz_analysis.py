import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

def plot_graph_on_img(image: np.ndarray, pos: np.ndarray, adjacency: np.ndarray):
    img = image.copy()
    adjacency_matrix = np.uint8(adjacency.copy())
    positions = pos.copy()
    pos_list = []
    for i in range(len(positions)):
        pos_list.append([positions[i][0], img.shape[0] - positions[i][1]])
    p = dict(enumerate(pos_list, 0))

    Graph = nx.from_numpy_matrix(adjacency_matrix)
    nx.set_node_attributes(Graph, p, 'pos')
    y_lim = img.shape[0]
    x_lim = img.shape[1]
    extent = 0, x_lim, 0, y_lim
    fig = plt.figure(frameon=False, figsize=(20, 20))
    plt.imshow(img, extent=extent, interpolation='nearest')
    nx.draw(Graph, pos=p, node_size=50, edge_color='g', width=3, node_color='r')
    plt.show()
    return fig


def plot_graph(image_size, pos: np.ndarray, adjacency: np.ndarray=None, nx_graph=None):

    if adjacency is not None:
        adjacency_matrix = np.uint8(adjacency.copy())
        Graph = nx.from_numpy_matrix(adjacency_matrix)
    else:
        Graph = nx_graph

    positions = pos.copy()
    pos_list = []
    ann = []
    for i in range(len(positions)):
        pos_list.append(
            [positions[i][0], image_size[0] - positions[i][1]])# flip y-axis, since node_pos are in image coordinates
        ann.append(str(i))
    p = dict(enumerate(pos_list, 0))
    nx.set_node_attributes(Graph, p, 'pos')
    y_lim = image_size[0]
    x_lim = image_size[1]
    extent = 0, x_lim, 0, y_lim
    fig = plt.figure(frameon=False, figsize=(20, 20))
    nx.draw(Graph, pos=p, node_size=50, edge_color='g', width=3, node_color='r')
    for i in range(0,np.shape(ann)[0],1):
        plt.annotate(ann[i], (pos_list[i][0], pos_list[i][1]))
    plt.show()
    return fig

def plot_graph2(image_size, pos: np.ndarray, adjacency: np.ndarray):

    adjacency_matrix = np.uint8(adjacency.copy())
    positions = pos.copy()
    pos_list = []
    for i in range(len(positions)):
        pos_list.append([positions[i][0],  positions[i][1]])
    p = dict(enumerate(pos_list, 0))

    Graph = nx.from_numpy_matrix(adjacency_matrix)
    nx.set_node_attributes(Graph, p, 'pos')
    y_lim = image_size[0]
    x_lim = image_size[1]
    extent = 0, x_lim, 0, y_lim
    fig = plt.figure(frameon=False, figsize=(20, 20))
    nx.draw(Graph, pos=p, node_size=50, edge_color='g', width=3, node_color='r')
    plt.show()
    return fig

def plot_nodes_on_img(image: np.ndarray, pos: np.ndarray, node_thick: int):
    img = image.copy()
    print(img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #positions = pos.astype(int)
    positions = pos
    for i in range(len(positions)):
        cv2.circle(img, (positions[i][0], positions[i][1]), 0, (255, 0, 0), node_thick)
    y_lim = img.shape[0]
    x_lim = img.shape[1]
    extent = 0, x_lim, 0, y_lim
    fig = plt.figure(frameon=False, figsize=(20, 20))
    plt.imshow(img, extent=extent, interpolation='nearest')
    plt.show()
    return img


def get_data(training_generator, batch_nr = 0, item_in_batch = 0):
    batch_image_rgb, batch_image_grey, batch_graph, batch_pos, batch_node_degrees, batch_node_container, batch_stacked_adj_matr = get_batched_data(training_generator, batch_nr= batch_nr)

    image_rgb = batch_image_rgb[item_in_batch]
    image_grey = batch_image_grey[item_in_batch]
    graph = batch_graph[item_in_batch]
    pos = batch_pos[item_in_batch]
    node_degrees = batch_node_degrees[item_in_batch]
    node_container = batch_node_container[item_in_batch]

    stacked_adj_matr =  batch_stacked_adj_matr[item_in_batch]
    adj_matrix = stacked_adj_matr[0]
    stacked_features = stacked_adj_matr[1:]

    return image_rgb, image_grey, graph, pos, node_degrees, node_container, adj_matrix, stacked_features


def get_batched_data(training_generator, batch_nr=0):
    image_rgb, image_grey, graph = training_generator.__getitem__(batch_nr=batch_nr)
    # graph element is build up in a list by the following structure ## [batch_resized_node_pos, batch_node_degrees, batch_node_types, batch_stacked_adj_matr]
    batch_image_rgb = image_rgb
    batch_image_grey = image_grey

    batch_graph = graph[0]
    batch_pos = graph[1]
    batch_node_degrees = graph[2]
    batch_node_container = graph[3]
    batch_stacked_adj_matr = graph[4]

    return batch_image_rgb, batch_image_grey, batch_graph, batch_pos, batch_node_degrees, batch_node_container, batch_stacked_adj_matr


def create_key_points(pos,size=1):
    import cv2
    kps=[]
    for i in range(len(pos)):
        kp=cv2.KeyPoint(x=pos[i,0].astype(float),y=pos[i,1].astype(float),_size=size)
        kps.append(kp)
    return kps


def load_data_sample(data_generator, desired_batch_nr, item_in_batch,  show_current_image=False):

    batch_object = data_generator.__getitem__(batch_nr=desired_batch_nr)
    image_rgb = batch_object[0][item_in_batch]
    image_grey = batch_object[1][item_in_batch]
    batch_graph = batch_object[2]
    batch_nx_graph= batch_graph[0]
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
    if show_current_image:
        fig = plot_graph_on_img(image_rgb, pos, adj_matrix)
    return image_rgb, image_grey, nx_graph, pos, node_degrees, node_container, adj_matrix, stacked_adj_features


def plot_2_graphs(image_size, pos1,erg: np.ndarray, pos2: np.ndarray, adjacency1: np.ndarray = None,
               adjacency2: np.ndarray = None, nx_graph=None,shw = True):
    fig, ax = plt.subplots()
    if adjacency1 is not None:
        adjacency_matrix = np.uint8(adjacency1.copy())
        Graph = nx.from_numpy_matrix(adjacency_matrix)
    else:
        Graph = nx_graph
    positions = pos1.copy()
    pos_list = []
    for i in range(len(positions)):
        pos_list.append(
            [positions[i][0], image_size[0] - positions[i][1]])  # flip y-axis, since node_pos are in image coordinates
    p1 = dict(enumerate(pos_list, 0))
    nx.set_node_attributes(Graph, p1, 'pos')
    y_lim = image_size[0]
    x_lim = image_size[1]
    extent = 0, x_lim, 0, y_lim
    # fig = plt.figure(frameon=False, figsize=(20, 20))
    nx.draw(Graph, pos=p1, node_size=50, edge_color='b', width=3, node_color='b', ax=ax)
    if adjacency2 is not None:
        adjacency_matrix = np.uint8(adjacency2.copy())
        Graph = nx.from_numpy_matrix(adjacency_matrix)
    else:
        Graph = nx_graph
    positions = pos2.copy()
    pos_list = []
    for i in range(len(positions)):
        pos_list.append(
            [positions[i][0], image_size[0] - positions[i][1]])  # flip y-axis, since node_pos are in image coordinates
    p2 = dict(enumerate(pos_list, 0))
    nx.set_node_attributes(Graph, p2, 'pos')
    y_lim = image_size[0]
    x_lim = image_size[1]
    extent = 0, x_lim, 0, y_lim
    fig = plt.figure(frameon=False, figsize=(20, 20))
    nx.draw(Graph, pos=p2, node_size=50, edge_color='r', width=3, node_color='r', ax=ax)
    if shw:
        ax.add_patch(ptc.Ellipse((erg[0][0], image_size[0] -erg[0][1]), 6, 6,0, color="black"))
        ax.add_patch(ptc.Ellipse((erg[1][0], image_size[0] -erg[1][1]), 6, 6, 0,color="orange"))
        ax.add_patch(ptc.Ellipse((erg[2][0], image_size[0] -erg[2][1]), 6, 6,0, color="green"))
        ax.add_patch(ptc.Ellipse((erg[3][0], image_size[0] -erg[3][1]), 6, 6, 0,color="pink"))
    plt.show()
    return fig
