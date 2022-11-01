import numpy as np
import networkx as nx
import cv2
import torch
from scipy.spatial import ConvexHull
from configs.plot.config_plots import *

def generate_mask(node_pos, mask_size, return_as_torch=True, device= 'cpu'):

    hull = ConvexHull(node_pos)
    # plt.plot(points[:, 0], points[:, 1], 'o')
    # for simplex in hull.simplices:
    #     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    # plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
    # plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')
    # plt.show()
    mask = np.zeros(mask_size)
    cv2.fillPoly(mask, pts=[np.array(node_pos[hull.vertices])], color=(255))
    mask = np.asarray(mask, np.uint8)
    if return_as_torch:
        return torch.from_numpy(mask).to(device=device).type(dtype=torch.bool)[None]
    else:
        return mask

def plot_image_with_mask(image, mask, is_grey=False):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    if is_grey:
        img_grey = image
        img_grey_rgb_channel = np.stack((img_grey, img_grey, img_grey), axis=2)
        image = img_grey_rgb_channel
    masked_image = np.full(image.shape, (255, 255, 255))
    masked_image = np.asarray(masked_image, np.uint8)
    masked_image[mask[0]] = image[mask[0]]
    #masked_image = cv2.bitwise_or(image, image, mask=mask)
    fig, ax = generate_mpl_figure(size_width=3.25, size_height=3.25)

    ax.imshow(masked_image)
    fig.show()

# test function
def nodes_in_mask(node_pos, mask):
    allnodes = node_pos
    not_inside = []
    all_inside = True
    for node in allnodes:
        if mask[node[1], node[0]] < 128:
            not_inside.append(node)
    if len(not_inside) > 0: all_inside = False

    return all_inside