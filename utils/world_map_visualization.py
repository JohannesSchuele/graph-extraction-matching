# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from configs.plot.config_plots import *
from configs.plot.colours import *
import torch
from pytorch3d.renderer import (
    PerspectiveCameras
)
from matplotlib import cm
import matplotlib as mpl

def create_world_map_mesh_and_graph_figure(world_point_cloud_map, mesh_model, fig=None, ax=None, size_width=3.5, size_height=3.5):

    if fig is None or ax is None:
        fig, ax = generate_mpl_3D_figure(size_width=size_width, size_height=size_height)

    #fig, ax = plot_mesh(mesh_model, fig=fig, ax=ax)
    # Get node positions
    pos = world_point_cloud_map.points_packed().clone().cpu().detach().numpy()

    # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
    # Those two points are the extrema of the line to be plotted
    for i, j in enumerate(world_point_cloud_map.nx_graph.edges()):
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))
        # Plot the connecting lines
        ax.plot(x, y, z, color=black, alpha=0.8, linewidth=0.75*SCALE_FIGURE_SETTINGs)


    # Get number of nodes
    n = world_point_cloud_map.nx_graph.number_of_nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([world_point_cloud_map.nx_graph.degree(i) for i in range(n)])

    # Define color range proportional to number of edges adjacent to a single node
    colors = [plt.cm.plasma(world_point_cloud_map.nx_graph.degree(i) / edge_max) for i in range(n)]

    # Loop on the pos dictionary to extract the x,y,z coordinates of each node
    for key, value in enumerate(pos):
        xi = value[0]
        yi = value[1]
        zi = value[2]
        # Scatter plot
        ax.scatter(xi, yi, zi, color=colors[key], s=1.5*SCALE_FIGURE_SETTINGs + 1*world_point_cloud_map.nx_graph.degree(key),
                   linewidths=0.3*SCALE_FIGURE_SETTINGs, edgecolors=black, alpha=0.9)



    return fig, ax

def plot_mesh(mesh_model,fig=None, ax=None,  reference_mesh4color=None, focus_on=None,
              cut_off_plane=None, iteration=None,
              color_nodes=COLOR_MESH, alpha_nodes =0.6, thickness_nodes=0.15*SCALE_FIGURE_SETTINGs,
              color_edges=COLOR_MESH, alpha_edges =0.4, linewidth=0.35*SCALE_FIGURE_SETTINGs):
    """
    Args:
        mesh_model: mesh to plot
        reference_mesh4color: if mesh is given, the ration of lengthening/contraction for every corresponding edge is
                              calculated and its color adjusted accordingly
        focus_on: One of 'retraction', 'expansion', None - will either use min, max or both values of edge length change
                  to determin colorbar of plot
        cut_off_plane: Tupel of torch Tensors (origin of normal vector, normal vector) describing the plane above which
                       (in direction of the normal vector) the vertices are considered for plotting
    Returns: Plot of mesh
    """

    # 3D network plot
    if fig is None and ax is None:
        fig, ax = generate_mpl_3D_figure()

    # mesh vertices and edges
    vertex_pos = mesh_model.verts_packed().clone().cpu().detach()
    mesh_edges = mesh_model.edges_packed().clone().cpu().detach()

    # prepare plot
    if reference_mesh4color is not None:
        # edge lengths of mesh
        v0, v1 = vertex_pos[mesh_edges].unbind(dim=1)
        edges_lengths = (v0 - v1).norm(dim=1, p=2)
        # edge lengths of reference mesh
        vertex_pos_ref = reference_mesh4color.verts_packed().clone().cpu().detach()
        mesh_edges_ref = reference_mesh4color.edges_packed().clone().cpu().detach()
        v0, v1 = vertex_pos_ref[mesh_edges_ref].unbind(dim=1)
        edges_lengths_ref = (v0 - v1).norm(dim=1, p=2)

        # # Functions to be used in color mapping 2 visualise how many times larger or smaller the edge is
        # # num of times smaller (negative) or larger (positive) + 1 or -1 to center around zero
        # _forward = np.vectorize((lambda val: val - 1 if val >= 1 else -1*(1/val) + 1), otypes=[float])
        # # inverse of previous function
        # _backward = np.vectorize((lambda val: val + 1 if val >= 0 else -1/(val - 1)), otypes=[float])
        # norm_color = mpl.colors.FuncNorm((_forward, _backward), vmin=v_min, vmax=v_max)

        # calculate color based on how much longer/shorter edge is than previously
        edge_length_features = edges_lengths/edges_lengths_ref
        if focus_on == 'retraction':
            v_max = 1/edge_length_features.min()
        elif focus_on == 'expansion':
            v_max = edge_length_features.max()
        else:
            v_max = 1/edge_length_features.min() if 1/edge_length_features.min() > edge_length_features.max()\
                else edge_length_features.max()  # For having 1 in center of logarithmic colour scale
        v_max = v_max if v_max != 1 else 2  # set to 2 if all values are 1 (first iteration)
        norm_color = mpl.colors.LogNorm(vmin=1/v_max, vmax=v_max)
        colormap = cm.get_cmap('seismic', 100)
        colors = colormap(norm_color(edge_length_features))
        fig.colorbar(cm.ScalarMappable(norm=norm_color, cmap=colormap), ax=ax)
        # change vertices color
        color_nodes = indigo  # Temp
    else:
        colors = np.zeros((len(mesh_edges), 4))  # dummy list

    # don't plot edges below cut_off_plane
    if cut_off_plane is not None:
        vertex_pos2plane_origin = vertex_pos - cut_off_plane[0]
        vertex_distance2plane = torch.matmul(vertex_pos2plane_origin, cut_off_plane[1])
        vertices_mask = torch.greater(vertex_distance2plane, 0)
        edges_mask = vertices_mask[mesh_edges]
        edges_mask = torch.logical_and(edges_mask[:, 0], edges_mask[:, 1])  # Todo: save masks
    else:  # if no plane given plot everything
        edges_mask = torch.full((mesh_edges.shape[0],), True)
        vertices_mask = torch.full((vertex_pos.shape[0],), True)

    # plot vertices and edges
    for mesh_edge, color in zip(mesh_edges[edges_mask, :], colors[edges_mask, :]):
        x = np.array((vertex_pos[mesh_edge[0]][0], vertex_pos[mesh_edge[1]][0]))
        y = np.array((vertex_pos[mesh_edge[0]][1], vertex_pos[mesh_edge[1]][1]))
        z = np.array((vertex_pos[mesh_edge[0]][2], vertex_pos[mesh_edge[1]][2]))
        # Plot the connecting lines
        if reference_mesh4color is not None:
            ax.plot(x, y, z, color=color, alpha=alpha_edges, linewidth=linewidth)
        else:
            ax.plot(x, y, z, color=color_edges, alpha=alpha_edges, linewidth=linewidth)
    vertex_pos_thinned = vertex_pos[vertices_mask, :].numpy()
    ax.scatter(vertex_pos_thinned[:, 0], vertex_pos_thinned[:, 1], vertex_pos_thinned[:, 2],
               color=color_nodes, alpha=alpha_nodes, s=thickness_nodes/2)

    return fig, ax


def camera_from_list_to_batch(camera_list):
    device = camera_list[0].device
    with torch.no_grad():
        R_batch = torch.zeros((camera_list.__len__(), 3, 3), dtype=torch.float32, device=device)
        T_batch = torch.zeros((camera_list.__len__(), 3), dtype=torch.float32, device=device)
        for i, camera in enumerate(camera_list):
            graph_image = camera
            R_batch[i, :] = camera.R
            T_batch[i, :] = camera.T
            # if i == 0:
            #     # furthermore, we know that the first camera is a trivial one
            #     self.r_rotation_init = graph_image.r_rotation.clone().detach()
            #     self.t_translation_init = graph_image.t_translation.clone().detach()


        assert T_batch.ndim == 2
        focal_length = None
        # cameras = PerspectiveCameras(focal_length=focal_length, R=R_batch.detach(), T=T_batch.detach(),
        #                              device=device)
        cameras = PerspectiveCameras(R=R_batch.detach(), T=T_batch.detach(),
                                     device=device)
        # ToDo delete the following!
        # plot_camera_scene(cameras,cameras_gt=cameras,status='camera poses')
    return cameras

