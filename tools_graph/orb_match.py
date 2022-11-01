import numpy as np
from tools_graph.match_orb_tools import BRIEF, match
import matplotlib.pyplot as plt
from skimage.feature import plot_matches
from tools_graph.plot_functs import plot_graph_matches22, plot_graph_matches2
from skimage.transform import warp, AffineTransform, EssentialMatrixTransform, FundamentalMatrixTransform, SimilarityTransform, PolynomialTransform, EuclideanTransform
from skimage.measure import ransac

import copy
def match_orb(img1, img2,pos1, pos2, adj1, adj2, plot = True):
    #d1 = BRIEF(img1,np.flip(copy.deepcopy(pos1),1), mode='uniform', patch_size=50, n=250)
    #d2 = BRIEF(img2, np.flip(copy.deepcopy(pos2),1), mode='uniform', patch_size=50, n=250)
    #d1 = BRIEF(img1, copy.deepcopy(pos1),orientations = None, n = 256, patch_size = 18, sigma = 1, mode = 'uniform', sample_seed = 42)
    #d2 = BRIEF(img2, copy.deepcopy(pos2), orientations = None, n = 256, patch_size = 18, sigma = 1, mode = 'uniform', sample_seed = 42)
    d1 = BRIEF(img1, copy.deepcopy(pos1),orientations = None, n = 512, patch_size = 25, sigma = 0.1, mode = 'uniform', sample_seed = 42)
    d2 = BRIEF(img2, copy.deepcopy(pos2), orientations = None, n = 512, patch_size = 25, sigma = 0.1, mode = 'uniform', sample_seed = 42)
    matches = match(d1, d2, cross_check=True, max_distance=np.inf)
    if plot:
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)
        plot_matches(ax, img1, img2, np.flip(pos1,1), np.flip(pos2,1),  copy.deepcopy(matches))
        plt.show()
        fig3 = plot_graph_matches22(img1, img2,copy.deepcopy(pos1), adj1,copy.deepcopy(pos2), adj2,matches, color=None)
        plt.show()
        # fig4 = plot_graph_matches22(img1, img2,np.flip(copy.deepcopy(pos1),1), adj1,np.flip(copy.deepcopy(pos2),1), adj2,matches, color=None)
        # plt.show()
        pos1
    src = np.array(pos1[matches[:, 0]])
    dst = np.array(pos2[matches[:, 1]])
    #src = np.flip(src, 1)
    #dst = np.flip(dst, 1)
    # estimate affine transform model using all coordinates
    model = AffineTransform()
    # model.estimate(src, dst)
    model_essential = EssentialMatrixTransform()
    model_essential.estimate(src, dst)
    #model = model_essential
    # robustly estimate affine transform model with RANSAC
    model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                                   residual_threshold=4, max_trials=50)
    # model_essential, inliers = ransac((src, dst), EuclideanTransform, min_samples=3, residual_threshold=5,
    #                                   max_trials=100)
    #model_robust = model_essential
    outliers = inliers == False
    # print("Affine transform:")
    # print(f'Scale: ({model.scale[0]:.4f}, {model.scale[1]:.4f}), '
    #       f'Translation: ({model.translation[0]:.4f}, '
    #       f'{model.translation[1]:.4f}), '
    #       f'Rotation: {model.rotation:.4f}')
    # print("RANSAC:")
    # print(f'Scale: ({model_robust.scale[0]:.4f}, {model_robust.scale[1]:.4f}), '
    #       f'Translation: ({model_robust.translation[0]:.4f}, '
    #       f'{model_robust.translation[1]:.4f}), '
    #       f'Rotation: {model_robust.rotation:.4f}')
    if plot:
        print("Affine transform:")
        print(f'Scale: ({model.scale[0]:.4f}, {model.scale[1]:.4f}), '
              f'Translation: ({model.translation[0]:.4f}, '
              f'{model.translation[1]:.4f}), '
              f'Rotation: {model.rotation:.4f}')
        print("RANSAC:")
        print(f'Scale: ({model_robust.scale[0]:.4f}, {model_robust.scale[1]:.4f}), '
              f'Translation: ({model_robust.translation[0]:.4f}, '
              f'{model_robust.translation[1]:.4f}), '
              f'Rotation: {model_robust.rotation:.4f}')
        fig, ax = plt.subplots(nrows=2, ncols=1)
        plt.gray()
        inlier_idxs = np.nonzero(inliers)[0]
        plot_matches(ax[0], img1, img2, src, dst,
                     np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
        ax[0].axis('off')
        ax[0].set_title('Correct correspondences')
        outlier_idxs = np.nonzero(outliers)[0]
        plot_matches(ax[1], img1, img2, src, dst,
                     np.column_stack((outlier_idxs, outlier_idxs)), matches_color='r')
        ax[1].axis('off')
        ax[1].set_title('Faulty correspondences')
        plt.show()
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)
        plot_matches(ax, img1, img2, np.flip(copy.deepcopy(pos1), 1), np.flip(copy.deepcopy(pos2), 1), matches[inliers]) #ToDo implement deepcopy in the actual matching functions
        plt.show()
        #fig2 = plot_graph_matches2(img1, img2, copy.deepcopy(pos1), adj1, copy.deepcopy(pos2), adj2, matches[inliers], color=None)
        fig2 = plot_graph_matches22(img1, img2, copy.deepcopy(pos1), adj1, copy.deepcopy(pos2), adj2,
                             matches[inliers], color=None)
        plt.show()
    return matches[inliers]