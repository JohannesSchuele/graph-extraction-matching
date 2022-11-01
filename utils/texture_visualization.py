
import torch
from configs.plot.config_plots import *
from configs.plot.colours import *
from utils.world_map_visualization import plot_mesh
from utils.camera_visualization import plot_cameras


def plot_point_cloud_on_mesh(geometry_mesh, point_cloud_observed, point_cloud_target_mesh, fig=None, ax=None, camera=None):

    # point_cloud_target_mesh = self.global_point_patterns_form_textured_mesh
    # point_cloud_observed = texture_points_3d_from_observed_target.points_packed()
    if fig is None and ax is None:
        fig, ax = generate_mpl_3D_figure()
    if camera is not None:
        cameras = camera
        handle_cam = plot_cameras(ax, cameras, color=blue_5)
    fig, ax = plot_mesh(geometry_mesh, fig=fig, ax=ax, alpha_nodes=0.4, alpha_edges=0.3, thickness_nodes=0.35,
                        color_nodes=COLOR_MESH, color_edges=COLOR_MESH, linewidth=0.35)
    point_cloud_target_mesh_np = point_cloud_target_mesh.detach().cpu().numpy()
    point_cloud_observed_np = point_cloud_observed.detach().cpu().numpy()

    for key, value in enumerate(point_cloud_target_mesh_np):
        xi = value[0]
        yi = value[1]
        zi = value[2]
        # Scatter plot
        ax.scatter(xi, yi, zi, color=COLOR_STITCHES_3D_MAP, s=1.0, alpha=1,
                   linewidths=0.6 * SCALE_FIGURE_SETTINGs, edgecolors=blue_2)

    for key, value in enumerate(point_cloud_observed_np):
        xi = value[0]
        yi = value[1]
        zi = value[2]
        # Scatter plot
        ax.scatter(xi, yi, zi, color=COLOR_STITCHES_3D_MAP, s=1.0, alpha=1,
                   linewidths=0.6 * SCALE_FIGURE_SETTINGs, edgecolors=red_1)  #red_2
    # ax.grid(False)
    ax.axis(False)
    ax.view_init(elev=90, azim=90) # elev to decentralize the view
    fig.show()
    return fig, ax



def visualize_mesh(textured_mesh, renderer, camera_perspective, title='',
                         silhouette=False, grey =False, size_width=3.5, size_height=3.5):
    inds = 3 if silhouette else range(3)
    #inds = 0 if grey else range(3)
    with torch.no_grad():
        predicted_images = renderer(textured_mesh, cameras=camera_perspective)
        img = predicted_images.detach().cpu().numpy()[0, :, :, :3]

    fig, ax = generate_mpl_figure(size_width=size_width, size_height=size_height)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax

def visualize_texture_prediction(textured_mesh, renderer, camera_perspective,
                         image_to_compare_torch, title='',
                         silhouette=False, grey = False):
    inds = 3 if silhouette else range(3)
    #inds = 3 if grey else range(3)
    with torch.no_grad():
        predicted_images = renderer(textured_mesh, cameras=camera_perspective)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(image_to_compare_torch.cpu().detach().numpy())

    plt.title(title)
    plt.axis("off")
    plt.show()


def visualize_features_on_image(image, pos1: np.ndarray =None, pos2: np.ndarray = None,
                                node_thickness:int=20, title:str='Features on Image', size_width=7.5, size_height=7.5):
    if torch.is_tensor(image):
        img = image.detach().cpu().numpy()[0]

    else:
        img = image  # needs to be a numpy array!

    fig, ax = generate_mpl_figure(size_width=size_width, size_height=size_height)
    ax.imshow(img)

    if torch.is_tensor(pos1):
        pos1 = pos1.detach().cpu().numpy()
    if pos1 is not None:
        ax.scatter(x=pos1[:, 1], y=pos1[:, 0], s=node_thickness, c=blue_2, alpha=0.7)
    if torch.is_tensor(pos2):
        pos2 = pos2.detach().cpu().numpy()
    if pos2 is not None:
        ax.scatter(x=pos2[:, 1], y=pos2[:, 0], s=node_thickness, c=red_1, alpha=0.7)

    ax.axis('on')
    ax.set_title(title)
    fig.tight_layout()
    #fig.show()

    return fig, ax



# def visualize_image(img, size_width=3.75, size_height=3.75, title = ''):
#     if torch.is_tensor(img):
#         img = img[0].clone().cpu().detach().numpy()
#     fig, ax = generate_mpl_figure(size_width=size_width, size_height=size_height)
#     ax.imshow(img)
#     ax.axis('on')
#     ax.set_title(title)
#     fig.tight_layout()
#     fig.show()
#     return fig, ax

def visualize_diff_off_images(img1, img2, size_width=3.75, size_height=3.75, title = ''):
    fig, ax = generate_mpl_figure(size_width=size_width, size_height=size_height)
    ax.imshow(img1, alpha=0.5) #ToDo visualize difference of images by putting color values respectively!
    ax.imshow(img2, alpha=0.5)
    ax.axis('on')
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax