import copy
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#def plot_graph_matches_tmp(img1, img2, color=None)
def plot_graph_matches(img1, img2, pos1: np.ndarray, adj1: np.ndarray, pos2: np.ndarray, adj2: np.ndarray, match,  color=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2 #columns (x direction 2nd)
    adj1= np.uint8(adj1.copy())
    adj2= np.uint8(adj2.copy())
    pos1 = pos1.copy()
    pos2 = pos2.copy()
    #positions = pos
    pos_list1 = []
    pos_list1_tmp = []
    for i in range(len(pos1)):
        pos_list1.append([pos1[i][0], img1.shape[0] - pos1[i][1]])
        pos_list1_tmp.append([pos1[i][0],  pos1[i][1]])
    p1 = dict(enumerate(pos_list1, 0))
    p1_tmp = dict(enumerate(pos_list1_tmp, 0))
    pos_list2 = []
    pos_list2_tmp = []
    for i in range(len(pos2)):
        pos_list2.append([pos2[i][0]+img1.shape[0], img2.shape[0]-pos2[i][1]])
        pos_list2_tmp.append([pos2[i][0]+img1.shape[0], pos2[i][1]])
    p2 = dict(enumerate(pos_list2, 0))
    p2_tmp = dict(enumerate(pos_list2_tmp, 0))
    Graph1 = nx.from_numpy_matrix(adj1)
    nx.set_node_attributes(Graph1, p1, 'pos')
    Graph2 = nx.from_numpy_matrix(adj2)
    nx.set_node_attributes(Graph2, p2, 'pos')
    #matchesd = dict(enumerate(matches, 0))
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    ## https://gist.github.com/isker/11be0c50c4f78cad9549
    ## https://github.com/iago-suarez/beblid-opencv-demo/blob/main/demo.ipynb
    r = 2
    thickness = 1
    for m in match:
        if color is None:
            # c = np.random.randint(0, 256, 3) if len(img1.shape) == 3 else np.random.randint(0, 256).tolist()
            # c = [17, 100, 150]
            c = np.random.randint(0, 256, 3).tolist()
        else:
            c = color
        # Generate random color for RGB/BGR and grayscale images as needed.
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(p1_tmp.get(m[0]))
        end2 = tuple(p2_tmp.get(m[1]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    fig = plt.figure(figsize=(20,10))

    nx.draw(Graph1, pos=p1, node_size=50, edge_color='g', width=3, node_color='r')
    nx.draw(Graph2, pos=p2, node_size=50, edge_color='g', width=3, node_color='r')
    y_lim = new_img.shape[0]
    x_lim = new_img.shape[1]
    extent = 0, x_lim, 0, y_lim
    extent = 0, x_lim, 0, y_lim
    plt.imshow(new_img,extent=extent, interpolation='nearest')
    plt.show()


def plot_graph_on_img(image: np.ndarray, pos: np.ndarray, adjacency: np.ndarray):
    img = image.copy()
    if len(img.shape) == 2:
       img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    adjacency_matrix = np.uint8(adjacency.copy())
    positions = pos.copy()
    #positions = pos
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
    return fig



def plot_graph_matches2(img1, img2, pos1: np.ndarray, adj1: np.ndarray, pos2: np.ndarray, adj2: np.ndarray, match,  color=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    pos1 = np.flip(copy.deepcopy(pos1),1)
    pos2 = np.flip(copy.deepcopy(pos2),1)
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2 #columns (x direction 2nd)
    adj1= np.uint8(adj1.copy())
    adj2= np.uint8(adj2.copy())
    pos1 = pos1.copy()
    pos2 = pos2.copy()
    pos_list1 = []
    for i in range(len(pos1)):
        pos_list1.append([pos1[i][0], img1.shape[0] - pos1[i][1]])
    p1 = dict(enumerate(pos_list1, 0))
    pos_list2 = []
    pos_list2_tmp = []
    for i in range(len(pos2)):
        pos_list2.append([pos2[i][0]+img1.shape[0], img2.shape[0]-pos2[i][1]])
    p2 = dict(enumerate(pos_list2, 0))
    Graph1 = nx.from_numpy_matrix(adj1)
    nx.set_node_attributes(Graph1, p1, 'pos')
    Graph2 = nx.from_numpy_matrix(adj2)
    nx.set_node_attributes(Graph2, p2, 'pos')
    G_match = nx.Graph()
    for i,single_match in enumerate(match):
        G_match.add_node(2*i,pos=pos_list1[single_match[0]])
        G_match.add_node(2*i+1,pos=pos_list2[single_match[1]])
        G_match.add_edge(2*i,2*i+1)
    pos_match = nx.get_node_attributes(G_match, 'pos')
    fig = plt.figure(figsize=(20,10))
    nx.draw(Graph1, pos=p1, node_size=50, edge_color='g', width=3, node_color='r')
    nx.draw(Graph2, pos=p2, node_size=50, edge_color='g', width=3, node_color='r')
    nx.draw(G_match , pos =pos_match, node_size=30, edge_color='y', width=2, node_color='b')
    y_lim = new_img.shape[0]
    x_lim = new_img.shape[1]
    extent = 0, x_lim, 0, y_lim
    extent = 0, x_lim, 0, y_lim
    plt.imshow(new_img,extent=extent, interpolation='nearest')
    plt.show()
    return fig

def plot_graph_matches22(img1, img2, pos1: np.ndarray, adj1: np.ndarray, pos2: np.ndarray, adj2: np.ndarray, match,
                        color=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    matcht= copy.deepcopy(match)
    pos1t = copy.deepcopy(pos1)
    pos2t = copy.deepcopy(pos2)
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] =img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] =img2  # columns (x direction 2nd)
    adj1 = np.uint8(adj1.copy())
    adj2 = np.uint8(adj2.copy())
    pos1 = pos1.copy()
    pos2 = pos2.copy()
    pos_list1 = []
    for i in range(len(pos1t)):
        pos_list1.append([pos1t[i][0], img1.shape[0] - pos1t[i][1]])
    p1 = dict(enumerate(pos_list1, 0))
    pos_list2 = []
    pos_list2_tmp = []
    for i in range(len(pos2t)):
        pos_list2.append([pos2t[i][0] + img1.shape[0], img2.shape[0] - pos2t[i][1]])
    p2 = dict(enumerate(pos_list2, 0))
    Graph1 = nx.from_numpy_matrix(adj1)
    nx.set_node_attributes(Graph1, p1, 'pos')
    Graph2 = nx.from_numpy_matrix(adj2)
    nx.set_node_attributes(Graph2, p2, 'pos')
    G_match = nx.Graph()
    for i, single_match in enumerate(matcht):
        G_match.add_node(2 * i, pos=pos_list1[single_match[0]])
        G_match.add_node(2 * i + 1, pos=pos_list2[single_match[1]])
        G_match.add_edge(2 * i, 2 * i + 1)
    pos_match = nx.get_node_attributes(G_match, 'pos')
    fig = plt.figure(figsize=(20, 10))
    nx.draw(Graph1, pos=p1, node_size=50, edge_color='g', width=3, node_color='r')
    nx.draw(Graph2, pos=p2, node_size=50, edge_color='g', width=3, node_color='r')
    nx.draw(G_match, pos=pos_match, node_size=30, edge_color='y', width=2, node_color='b')
    y_lim = new_img.shape[0]
    x_lim = new_img.shape[1]
    extent = 0, x_lim, 0, y_lim
    extent = 0, x_lim, 0, y_lim
    plt.imshow(new_img, extent=extent, interpolation='nearest')
    plt.show()
    return fig

def plot_graph_matches_color(img1, img2, pos1: np.ndarray, adj1: np.ndarray, pos2: np.ndarray, adj2: np.ndarray, match,
                         color_edge=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    matcht = copy.deepcopy(match)
    pos1t = copy.deepcopy(pos1)
    pos2t = copy.deepcopy(pos2)
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2  # columns (x direction 2nd)
    adj1 = np.uint8(adj1.copy())
    adj2 = np.uint8(adj2.copy())
    pos1 = pos1.copy()
    pos2 = pos2.copy()
    pos_list1 = []
    for i in range(len(pos1t)):
        pos_list1.append([pos1t[i][0], img1.shape[0] - pos1t[i][1]])
    p1 = dict(enumerate(pos_list1, 0))
    pos_list2 = []
    pos_list2_tmp = []
    for i in range(len(pos2t)):
        pos_list2.append([pos2t[i][0] + img1.shape[0], img2.shape[0] - pos2t[i][1]])
    p2 = dict(enumerate(pos_list2, 0))
    Graph1 = nx.from_numpy_matrix(adj1)
    nx.set_node_attributes(Graph1, p1, 'pos')
    Graph2 = nx.from_numpy_matrix(adj2)
    nx.set_node_attributes(Graph2, p2, 'pos')
    G_match = nx.Graph()
    color_map_edge = []
    color_map_node = []
    for i, single_match in enumerate(matcht):
        G_match.add_node(2 * i, pos=pos_list1[single_match[0]])
        G_match.add_node(2 * i + 1, pos=pos_list2[single_match[1]])
        G_match.add_edge(2 * i, 2 * i + 1)
        color = (np.random.rand(3))
        color_map_edge.append(color)
        color_map_node.extend([color,color])
    pos_match = nx.get_node_attributes(G_match, 'pos')
    fig = plt.figure(figsize=(20, 10))
    nx.draw(Graph1, pos=p1, node_size=55, edge_color='g', width=3, node_color='r')
    nx.draw(Graph2, pos=p2, node_size=55, edge_color='g', width=3, node_color='r')
    #nx.draw(G_match, pos=pos_match, node_size=30, edge_color=color_map, width=2, node_color='b')
    nx.draw(G_match, pos=pos_match, node_size=35, edge_color=color_map_edge, width=2, node_color=color_map_node)
    y_lim = new_img.shape[0]
    x_lim = new_img.shape[1]
    extent = 0, x_lim, 0, y_lim
    extent = 0, x_lim, 0, y_lim
    plt.imshow(new_img, extent=extent, interpolation='nearest')
    plt.show()
    return fig

def plot_graph_matches_color2(img1, img2, pos1: np.ndarray, adj1: np.ndarray, pos2: np.ndarray, adj2: np.ndarray, match,x1,y1,width1,height1,
                         x2,y2,width2,height2,color_edge=None):

    matcht = copy.deepcopy(match)
    pos1t = copy.deepcopy(pos1)
    pos2t = copy.deepcopy(pos2)
    fig, ax = plt.subplots()
    ax.add_patch(Rectangle((x1,img1.shape[0]-y1),width1,height1,color="yellow",fc ='none',linewidth=5,linestyle="dotted"))
    ax.add_patch(Rectangle((x2+ img1.shape[0], img2.shape[0] - y2), width2, height2, color="yellow",fc ='none',linewidth=5,linestyle="dotted"))
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2  # columns (x direction 2nd)
    adj1 = np.uint8(adj1.copy())
    adj2 = np.uint8(adj2.copy())
    pos1 = pos1.copy()
    pos2 = pos2.copy()
    pos_list1 = []
    for i in range(len(pos1t)):
        pos_list1.append([pos1t[i][0], img1.shape[0] - pos1t[i][1]])
    p1 = dict(enumerate(pos_list1, 0))
    pos_list2 = []
    pos_list2_tmp = []
    for i in range(len(pos2t)):
        pos_list2.append([pos2t[i][0] + img1.shape[0], img2.shape[0] - pos2t[i][1]])
    p2 = dict(enumerate(pos_list2, 0))
    Graph1 = nx.from_numpy_matrix(adj1)
    nx.set_node_attributes(Graph1, p1, 'pos')
    Graph2 = nx.from_numpy_matrix(adj2)
    nx.set_node_attributes(Graph2, p2, 'pos')
    G_match = nx.Graph()
    color_map_edge = []
    color_map_node = []
    for i, single_match in enumerate(matcht):
        G_match.add_node(2 * i, pos=pos_list1[single_match[0]])
        G_match.add_node(2 * i + 1, pos=pos_list2[single_match[1]])
        G_match.add_edge(2 * i, 2 * i + 1)
        color = (np.random.rand(3))
        color_map_edge.append(color)
        color_map_node.extend([color,color])
    pos_match = nx.get_node_attributes(G_match, 'pos')
    fig = plt.figure(figsize=(20, 10))
    nx.draw(Graph1, pos=p1, node_size=55, edge_color='g', width=3, node_color='r',ax=ax)
    nx.draw(Graph2, pos=p2, node_size=55, edge_color='g', width=3, node_color='r',ax=ax)
    #nx.draw(G_match, pos=pos_match, node_size=30, edge_color=color_map, width=2, node_color='b')
    nx.draw(G_match, pos=pos_match, node_size=35, edge_color=color_map_edge, width=2, node_color=color_map_node,ax=ax)
    y_lim = new_img.shape[0]
    x_lim = new_img.shape[1]
    extent = 0, x_lim, 0, y_lim
    extent = 0, x_lim, 0, y_lim
    plt.show()
    return fig

def plot_graph_matches_color_inliers_outliers(img1, img2, pos1: np.ndarray, adj1: np.ndarray, pos2: np.ndarray, adj2: np.ndarray,
                             matches_brute_force, inliers, title=None,
                             color_edge=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    matches_brute_force_t = copy.deepcopy(matches_brute_force)
    pos1t = copy.deepcopy(pos1)
    pos2t = copy.deepcopy(pos2)
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2  # columns (x direction 2nd)
    adj1 = np.uint8(adj1.copy())
    adj2 = np.uint8(adj2.copy())
    pos1 = pos1.copy()
    pos2 = pos2.copy()
    pos_list1 = []
    for i in range(len(pos1t)):
        pos_list1.append([pos1t[i][0], img1.shape[0] - pos1t[i][1]])
    p1 = dict(enumerate(pos_list1, 0))
    pos_list2 = []
    pos_list2_tmp = []
    for i in range(len(pos2t)):
        pos_list2.append([pos2t[i][0] + img1.shape[0], img2.shape[0] - pos2t[i][1]])
    p2 = dict(enumerate(pos_list2, 0))
    Graph1 = nx.from_numpy_matrix(adj1)
    nx.set_node_attributes(Graph1, p1, 'pos')
    Graph2 = nx.from_numpy_matrix(adj2)
    nx.set_node_attributes(Graph2, p2, 'pos')
    G_match = nx.Graph()
    color_map_edge = []
    color_map_node = []
    for i, single_match in enumerate(matches_brute_force_t[inliers]):
        G_match.add_node(2 * i, pos=pos_list1[single_match[0]])
        G_match.add_node(2 * i + 1, pos=pos_list2[single_match[1]])
        G_match.add_edge(2 * i, 2 * i + 1)
        color = (np.random.rand(3))
        color_map_edge.append(color)
        color_map_node.extend([color, color])
    pos_match = nx.get_node_attributes(G_match, 'pos')
    fig = plt.figure(figsize=(20, 10))
    nx.draw(Graph1, pos=p1, node_size=55, edge_color='g', width=5, node_color='r')
    nx.draw(Graph2, pos=p2, node_size=55, edge_color='g', width=5, node_color='r')
    # nx.draw(G_match, pos=pos_match, node_size=30, edge_color=color_map, width=2, node_color='b')
    nx.draw(G_match, pos=pos_match, node_size=60, edge_color=color_map_edge, width=2.5, node_color=color_map_node)

    G_match_outliers = nx.Graph()
    color_map_edge = []
    color_map_node = []
    for i, single_match in enumerate(matches_brute_force_t[~inliers]):
        G_match_outliers.add_node(2 * i, pos=pos_list1[single_match[0]])
        G_match_outliers.add_node(2 * i + 1, pos=pos_list2[single_match[1]])
        G_match_outliers.add_edge(2 * i, 2 * i + 1)
    pos_match = nx.get_node_attributes(G_match_outliers, 'pos')
    nx.draw(G_match_outliers, pos=pos_match, node_size=60, edge_color='r', width=5, node_color='r')

    y_lim = new_img.shape[0]
    x_lim = new_img.shape[1]
    extent = 0, x_lim, 0, y_lim

    plt.imshow(new_img, extent=extent, interpolation='nearest')
    if title is not None:
        plt.title(title)
    #plt.savefig('matching_grey3.eps', format='eps')
    plt.show()

    return fig