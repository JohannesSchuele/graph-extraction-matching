# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from pytorch3d.vis.plotly_vis import get_camera_wireframe
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    FoVPerspectiveCameras
)

def render_point_could_2image(point_cloud, camera, points_per_pixel=1, radius=0.009, znear=1, image_size =128, show_image = False):
    #camera_4_point_cloud = FoVOrthographicCameras(device=camera.device, R=camera.R, T=camera.T, znear=znear)
    camera_4_point_cloud = FoVPerspectiveCameras(device=camera.device, R=camera.R, T=camera.T, znear=znear)
    #camera_4_point_cloud = camera
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 128x128. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel
    )
    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=camera_4_point_cloud, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        # Pass in background_color to the alpha compositor, setting the background color
        # to the 3 item tuple, representing rgb on a scale of 0 -> 1, in this case blue
        compositor=AlphaCompositor(background_color=(1, 1, 1))
    )
    image_of_point_cloud = renderer(point_cloud)
    if show_image:
        plt.figure(figsize=(10, 10))
        plt.imshow(image_of_point_cloud[0, ..., :3].cpu().numpy())
        plt.axis("off");
        plt.show()
    return image_of_point_cloud

import warnings

def plot_pointcloud(point_cloud, title="Point Cloud"):
    # Sample points uniformly from the surface of the mesh.
    # ToDo:  Pass the keyword argument auto_add_to_figure in order to get rid of the warnings
    points = point_cloud.points_packed()
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
    #fig.add_axes(ax)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    #ax.view_init(0, 0)
    plt.show()


def plot_cameras(ax, cameras, color: str = "blue"):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    cam_wires_canonical = get_camera_wireframe().cuda()[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    return plot_handles


def plot_camera_scene(cameras, cameras_gt, status: str):
    """
    Plots a set of predicted cameras `cameras` and their corresponding
    ground truth locations `cameras_gt`. The plot is named with
    a string passed inside the `status` argument.
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.clear()
    ax.set_title(status)
    handle_cam = plot_cameras(ax, cameras, color="#FF7D1E")
    handle_cam_gt = plot_cameras(ax, cameras_gt, color="#812CE5")
    plot_radius = 3
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([3 - plot_radius, 3 + plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    labels_handles = {
        "Estimated cameras": handle_cam[0],
        "GT cameras": handle_cam_gt[0],
    }
    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
    )
    plt.show()
    return fig
