import os
import torch
import matplotlib.pyplot as plt

from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm.notebook import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj
import networkx as nx
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)


def create_data_set(input_mesh,num_views, device):
    # the number of different viewpoints from which we want to render the mesh.
    #num_views = 20
    # Get a batch of viewing angles.
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)
    # Place a point light in front of the object. As mentioned above, the front of
    # the cow is facing the -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    # Initialize an OpenGL perspective camera that represents a batch of different
    # viewing angles. All the cameras helper methods support mixed type inputs and
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and
    # then specify elevation and azimuth angles for each viewpoint as tensors.
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    # We arbitrarily choose one particular view that will be used to visualize
    # results
    camera_init = FoVPerspectiveCameras(device=device, R=R[None, 1, ...],
                                      T=T[None, 1, ...])
    # Define the settings for rasterization and shading. Here we set the output
    # image to be of size 128X128. As we are rendering images for visualization
    # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
    # rasterize_meshes.py for explanations of these parameters.  We also leave
    # bin_size and max_faces_per_bin to their default values of None, which sets
    # their values using heuristics and ensures that the faster coarse-to-fine
    # rasterization method is used.  Refer to docs/notes/renderer.md for an
    # explanation of the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=128,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured
    # Phong shader will interpolate the texture uv coordinates for each vertex,
    # sample from a texture image and apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera_init,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=camera_init,
            lights=lights
        )
    )

    # Create a batch of meshes by repeating the cow mesh and associated textures.
    # Meshes has a useful `extend` method which allows us do this very easily.
    # This also extends the textures.
    meshes = input_mesh.extend(num_views)
    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras, lights=lights)
    # Our multi-view cow dataset will be represented by these 2 lists of tensors,
    # each of length num_views.
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
    target_cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...],
                                               T=T[None, i, ...]) for i in range(num_views)]
    return cameras, camera_init, lights, meshes,target_images, target_rgb, target_cameras

## 3. Mesh prediction via silhouette rendering
# In the previous section, we created a dataset of images of multiple viewpoints of a cow.
# In this section, we predict a mesh by observing those target images without any knowledge of the ground truth cow mesh.
# We assume we know the position of the cameras and lighting.
# We first define some helper functions to visualize the results of our mesh prediction:
# Show a visualization comparing the rendered predicted mesh to the ground truth
# mesh

# visualize_prediction(predicted_mesh, renderer=renderer_silhouette,
#                          target_image=target_rgb[1], title='',
#                          silhouette=False)
def visualize_prediction(predicted_mesh, renderer,
                         target_image, title='',
                         silhouette=False):
    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())

    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_graph_vs_point_cloud(img : np.ndarray, pos: np.ndarray, adjacency: np.ndarray,image_2_compare: np.ndarray, title=''):
    adjacency_matrix = np.uint8(adjacency.copy())
    positions = pos.copy()
    # positions = pos
    pos_list = []
    for i in range(len(positions)):
        #pos_list.append([positions[i][0], img.shape[0] - positions[i][1]])
        pos_list.append([positions[i][0],  positions[i][1]]) #ToDo check why it gets resorted
    p = dict(enumerate(pos_list, 0))
    Graph = nx.from_numpy_matrix(adjacency_matrix)
    nx.set_node_attributes(Graph, p, 'pos')
    y_lim = img.shape[0]
    x_lim = img.shape[1]
    extent = 0, x_lim, 0, y_lim
    fig = plt.figure(frameon=False, figsize=(20, 20))
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img, extent=extent, interpolation='nearest')
    nx.draw(Graph, pos=p, node_size=50, edge_color='g', width=3, node_color='r')


    plt.subplot(1, 2, 2)
    plt.imshow(image_2_compare)
    plt.title(title)
    plt.axis("off")
    plt.show()


# Plot losses as a function of optimization iteration
def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l['values'], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")


def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)
    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)
    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")
