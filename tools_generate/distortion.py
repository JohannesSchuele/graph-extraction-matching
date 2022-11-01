import networkx as nx
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tools_graph.utilz_analysis import plot_nodes_on_img

# def get_camera_params_original():
#     h=128
#     w=128
#
#     # camera parameters: ret, mtx, dist, rvecs, tvecs
#     fx = 1  #1058
#     fy = 1  #1041
#     cx = w/2
#     cy = h/2
#
#     # distortion parameters
#     k1 = 0.0
#     k2 = 0.0
#     p1 = 0.00
#     p2 = 0.0
#
#     # convert for opencv
#     mtx = np.matrix([
#         [fx,  0, cx],
#         [ 0, fy, cy],
#         [ 0,  0,  1]
#     ], dtype = "float32")
#
#     dist = np.array([k1, k2, p1, p2], dtype = "float32")
#     cameraparams = [0, mtx, dist]
#     return cameraparams

def get_camera_params():
    h=128
    w=128

    # camera parameters: ret, mtx, dist, rvecs, tvecs
    fx = 1  #1058
    fy = 1  #1041
    cx = w/2
    cy = h/2

    # distortion parameters
    k1 = 0.000055
    k2 = 0.000000015
    p1 = 0.0000001
    p2 = -0.005

    # convert for opencv
    mtx = np.matrix([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype = "float32")

    dist = np.array([k1, k2, p1, p2], dtype = "float32")
    cameraparams = [0, mtx, dist]
    return cameraparams

def distort_image(cameraparams, image, input_pos, plot_it=False, adjacency_matrix=None):
    # plot undistorted graph-points on undistorted image

    image_size = image.shape[0]
    mtx = cameraparams[1]
    dist = cameraparams[2]
    #image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cvt = image/255
    undist_image_uncut = cv2.undistort(image_cvt, mtx, dist)

    if len(np.shape(undist_image_uncut)) > 2:
        (h, w, d) = np.shape(undist_image_uncut)
    else:
        (h, w) = np.shape(undist_image_uncut)
    dh = int(np.round(h / 2, 0) - image_size / 2)
    dw = int(np.round(w / 2, 0) - image_size / 2)
    undist_image = undist_image_uncut[dh:dh + image_size, dw:dw + image_size]

    # pos = nx.get_node_attributes(self.Graph, "pos")
    # pos_list = np.float32(get_position_vector(pos))
    input_pos_f32 = np.ascontiguousarray(input_pos, np.float32)
    undist_pos_centralized = cv2.undistortPoints(input_pos_f32, mtx, dist)[:, 0, :]
    undist_pos= undist_pos_centralized
    undist_pos[:, 0] = undist_pos_centralized[:, 0] + 0.5 * undist_image.shape[1]
    undist_pos[:, 1] = undist_pos_centralized[:, 1] + 0.5 * undist_image.shape[0]

    # #undist_pos = cv2.undistortPoints(input_pos_f32, mtx, dist)[:, 0, :]
    # pos = dict(enumerate(undist_pos, 0))
    # pos_dst_image = {}  # undistorted positions for plot
    # for key in pos.keys():
    #     pos_dst_image[key] = (undist_pos[key][0] + 0.0 * undist_image.shape[1],
    #                           undist_pos[key][1] - 0.5 * undist_image.shape[0])

    if plot_it:
        fig = plt.figure(frameon=False, figsize=(20, 20))
        #y_lim, x_lim = undist_image.shape[:-1]
        #extent = 0, x_lim, 0, -y_lim
        # plot function
        y_lim = undist_image.shape[0]
        x_lim = undist_image.shape[1]
        extent = 0, x_lim, 0, y_lim
    if plot_it and adjacency_matrix is not None:

        pos_dst_image = dict(enumerate(undist_pos, 0))
        fig = plt.figure(frameon=False)
        plt.imshow(undist_image, extent=extent)
        Graph = nx.from_numpy_matrix(adjacency_matrix)
        nx.draw(Graph, pos=pos_dst_image, node_size=5, edge_color='y', width=1, node_color='r')
        plt.show()
    elif plot_it and adjacency_matrix is None:
        plt.figure()
        # plot x and y using blue circle markers
        plt.plot(undist_pos[:, 0], image_size - undist_pos[:, 1], 'bo')
        plt.imshow(undist_image, extent=extent, interpolation='nearest')
        plt.title('optimized cam image in red, target cam image in blue')
        plt.show()

    return undist_pos, undist_image




