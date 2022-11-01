import torch
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from models.camera_pose import ProjectionModel, ProjectionModelIntrinsicCalibration
import random
from configs.plot.config_plots import PLOT_CALIBRATION_RANSAC
from configs.plot.config_plots import *
def triangulate_ransac(keypoints_3d, image_keypoints_2d,
                       r_rotation_init, t_translation_init,
                       camera,
                       image_size,
                       n_sample_iters=14, # ToDo: automize!
                       ransac_iters_per_samples=40,
                       reprojection_error_epsilon=1,
                       direct_optimization=True,
                       device='cpu'):
    assert len(keypoints_3d) == len(image_keypoints_2d)
    assert len(image_keypoints_2d) >= 4
    #reprojection_error_epsilon = 12

    n_views = len(image_keypoints_2d)
    # determine inliers
    view_set = set(range(n_views))
    inlier_set = set()
    for i in range(n_sample_iters):
        sampled_views = sorted(random.sample(view_set, 6))

        estimated_model = optimize_camera_pose_by_projection(keypoints_3d=keypoints_3d[sampled_views],
                                                                  image_keypoints_2d=image_keypoints_2d[sampled_views],
                                                                  r_rotation=r_rotation_init,
                                                             t_translation=t_translation_init,
                                                             camera=camera,
                                                                image_size =image_size,
                                                                  lr=0.01, iterations=ransac_iters_per_samples, device=device)
        r_rotation_estimated, t_translation_estimated, _ = estimated_model
        model_camera_projection = ProjectionModel(keypoints_3d=keypoints_3d,
                                                  image_keypoints_2d=image_keypoints_2d,
                                                  r_param_rotation=r_rotation_estimated, t_param_translation=t_translation_estimated,
                                                  camera=camera,
                                                  image_size=image_size,
                                                  device=device
                                                  )
        _, reprojection_error_residual, _ = model_camera_projection()
        reprojection_error_vector = torch.mean(reprojection_error_residual, dim=1)

        new_inlier_set = set(sampled_views)
        for view in view_set:  ##error vector needs to be calculated for all points
            current_reprojection_error = reprojection_error_vector[view]
            if PRINT_RANSAC_REPORJECTION_ERROR:
                print(current_reprojection_error)
            if current_reprojection_error < reprojection_error_epsilon:
                new_inlier_set.add(view)

        ## check for the best model
        if len(new_inlier_set) > len(inlier_set):
            inlier_set = new_inlier_set
            best_model = estimated_model

    # triangulate using inlier_set
    if len(inlier_set) == 0:
        inlier_set = view_set.copy()
    inlier_list = torch.tensor(sorted(inlier_set))
    # direct reprojection error minimization
    if direct_optimization:
        r_rotation_opt, t_translation_opt, _ = optimize_camera_pose_by_projection(
            keypoints_3d=keypoints_3d[inlier_list],
            image_keypoints_2d=image_keypoints_2d[inlier_list],
            r_rotation=r_rotation_init,
            t_translation=t_translation_init,
            camera=camera,
            image_size=image_size,
            lr=0.01, iterations=int(ransac_iters_per_samples*1.8),
            device=device)
    return r_rotation_opt, t_translation_opt, inlier_list

def optimize_camera_pose_by_projection(keypoints_3d, image_keypoints_2d, r_rotation, t_translation, camera, image_size, lr = 0.01, iterations = 500, device='cpu'):
    focal_length = torch.tensor(1).float()
    model_camera_projection = ProjectionModel(keypoints_3d=keypoints_3d, image_keypoints_2d=image_keypoints_2d,
                                              r_param_rotation=r_rotation, t_param_translation=t_translation, camera=camera, image_size=image_size, device=device)
    optimizer = torch.optim.Adam(model_camera_projection.parameters(), lr=lr)
    loop = tqdm(range(iterations))
    for i in loop:
        # if i == 0:
        #     print('first')
        optimizer.zero_grad()
        loss, _, _ = model_camera_projection()  # ToDo implement a stop criterium if loss is to low, else it would cause problems for the differentiation!!!
        loss.backward()  # ToDo: check retain_graph here! and probably just do it in the first call!
        # ToDo check if retain_graoh in the first call is sufficient, and makes everything more efficient
        optimizer.step()

        #print('Infos here!')
        a=torch.any(torch.isnan(loss)).cpu().numpy()
        b=torch.any(torch.isnan(keypoints_3d)).cpu().numpy()
        c=np.any(np.isnan(image_keypoints_2d))
        d=torch.any(torch.isnan(r_rotation)).cpu().numpy()
        e=torch.any(torch.isnan(t_translation)).cpu().numpy()
        problem = a or b or c or d or e
        #print('Iteration loop number :', i, ' with current loss: ', loss)
        if problem:
            print('Problem there are some nan values!!')


    r_rotation_opt, t_translation_opt = model_camera_projection.evaluate()
    _, error_residual, re_projected_image_keypoints_2d = model_camera_projection()
    r_rotation_opt.requires_grad = False
    t_translation_opt.requires_grad = False

    #if i == len(loop) - 1:
    if PLOT_CALIBRATION_RANSAC:
        plt.figure()
        # print('loss in current iteration: ', loss)
        i = 2
        img_cv = re_projected_image_keypoints_2d.clone().cpu().detach().int()
        plt.plot(img_cv[:, 0], 128 - img_cv[:, 1], 'ro')
        plt.plot(img_cv[i, 0], 128 - img_cv[i, 1], 'go')  # plot x and y using blue circle markers
        image_keypoints_2d = torch.from_numpy(image_keypoints_2d).to(device=device)
        img_p3d = image_keypoints_2d.clone().cpu().detach()
        plt.plot(img_p3d[:, 0], 128 - img_p3d[:, 1], 'bo')
        plt.plot(img_p3d[i, 0], 128 - img_p3d[i, 1], 'go')
        plt.title('cv cam in red, pytorch cam in blue')
        plt.show()

    # print('Initial rotation: ', r_rotation, ' initial translation: ', t_translation)
    # print('Solution of camera pose optimization rotation: ', r_rotation_opt, 'translation optimized: ', t_translation_opt)
    # print('Focal Length: ', focal_length)
    return r_rotation_opt, t_translation_opt, error_residual



def calibrate_intrinsic_camera_parameters(keypoints_3d, image_keypoints_2d, adjacency, r_rotation, t_translation, camera, image_size, lr = 0.25, iterations = 50, device='cpu'):
    model_camera_calibration = ProjectionModelIntrinsicCalibration(keypoints_3d=keypoints_3d,
                                              image_keypoints_2d=image_keypoints_2d,
                                                                   adjacency=adjacency,
                                              r_rotation=r_rotation, t_translation=t_translation, camera=camera,
                                                                    image_size=image_size,
                                                                    device=device
                                              )
    optimizer = torch.optim.Adam(model_camera_calibration.parameters(), lr=lr)
    loop = tqdm(range(iterations))
    for i in loop:
        # if i == 0:
        #     print('first')
        optimizer.zero_grad()
        loss, _ = model_camera_calibration()  # ToDo implement a stop criterium if loss is to low, else it would cause problems for the differentiation!!!
        loss.backward(
            retain_graph=True)  # ToDo: check retain_graph here! and probably just do it in the first call!
        # ToDo check if retain_graph in the first call is sufficient, and makes everything more efficient
        optimizer.step()
        #print('Iteration loop number :', i, ' with current loss: ', loss)
        focal_length_im, principal_point_x_im, principal_point_y_im = model_camera_calibration.evaluate()
        # print('Intermediate solution of focal length: ', focal_length_im, 'principal_point_x: ', principal_point_x_im,
        #       'principal_point_y: ', principal_point_y_im)
    focal_length, principal_point_x, principal_point_y = model_camera_calibration.evaluate()
    _, error_residual = model_camera_calibration()
    focal_length.requires_grad = False
    principal_point_x.requires_grad = False
    principal_point_y.requires_grad = False
    #print('Solution of focal length: ', focal_length, 'principal_point_x: ', principal_point_x, 'principal_point_y: ', principal_point_y)
    return focal_length, principal_point_x, principal_point_y

