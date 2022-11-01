import glob
import skimage.io as io
import skimage.transform as trans
import numpy as np
import pylab as plt



def square_image(img, random = None):
    """ Square Image
    Function that takes an image (ndarray),
    gets its maximum dimension,
    creates a black square canvas of max dimension
    and puts the original image into the
    black canvas's center
    If random [0, 2] is specified, the original image is placed
    in the new image depending on the coefficient,
    where 0 - constrained to the left/up anchor,
    2 - constrained to the right/bottom anchor
    """
    size = max(img.shape[0], img.shape[1])
    new_img = np.zeros((size, size),np.float32)
    ax, ay = (size - img.shape[1])//2, (size - img.shape[0])//2

    if random and not ax == 0:
        ax = int(ax * random)
    elif random and not ay == 0:
        ay = int(ay * random)

    new_img[ay:img.shape[0] + ay, ax:ax+img.shape[1]] = img
    return new_img


def reshape_image(img, target_size):
    """ Reshape Image
    Function that takes an image
    and rescales it to target_size
    """
    img = trans.resize(img,target_size)
    img = np.reshape(img,img.shape+(1,))
    img = np.reshape(img,(1,)+img.shape)
    return img

def normalize_mask(mask):
    """ Mask Normalization
    Function that returns normalized mask
    Each pixel is either 0 or 1
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask

def show_image(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


def create_graph_tensor(mask,tensor_size):
    """ Mask Normalization
    Function that returns normalized mask
    Each pixel is either 0 or 1
    """

    mask = np.asarray(mask)
    y_positions_label = mask[:, 0:2, 0]
    y_adjacency_label = mask[:, 2:, 0]


    pos = y_positions_label.astype(int)
    adj_dim = int((len(pos) * len(pos) -len(pos)) / 2)
    tensor_graph = np.zeros((tensor_size, tensor_size, adj_dim))

    for node_idx in range(len(pos)):
        adjacency_idx_vec = np.argwhere(y_adjacency_label[node_idx, :] == 1)
        for adjacency_idx in range(len(adjacency_idx_vec)):
            global_adj_idx_tuple = pos[adjacency_idx, :]
            idx_trivec = get_indices_trivec_adjacency(adj_dim,node_idx,adjacency_idx)
            tensor_graph[node_idx[0], node_idx[1], idx_trivec] = 1
    tensor_graph.astype(int)

    return tensor_graph


def get_indices_trivec_adjacency(adj_dim,node_idx,adjacency_idx):
    adj_tri_vec = np.triu_indices(adj_dim, k=1)
    idx_trivec  = np.argwhere(adj_tri_vec == [node_idx,adjacency_idx] or adj_tri_vec == [adjacency_idx,node_idx])
    return idx_trivec[0]