import os
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology
from skimage.morphology import skeletonize

def threshold(
    filtered_img: np.ndarray, do_save: bool, filepath: str = ""
) -> np.ndarray:
    blur_kernel = (5, 5)
    blurred_img = cv2.GaussianBlur(filtered_img, blur_kernel, 0)
    _, thresholded_img = cv2.threshold(
        blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if do_save:
        cv2.imwrite(filepath, thresholded_img)

    return thresholded_img

def skeletonise_and_clean(
    thr_image: np.ndarray, plot: bool, save: bool, directory: str, edgelength: int=1,
):
    """
    Creates skeletonised image
    :param thr_image: thresholded image
    :param plot:
    :param save:
    :param directory:
    :return:
    """
    img= thr_image.astype("uint8")
    #edgelength = 3

    thr_image = threshold(filtered_img= img, do_save = False, filepath = "")
    img = thr_image.copy() / 255
    skeleton_noisy_bool = skeletonize(img)  #.astype(int)*255

    # remove too small edges
    labeled = morphology.label(skeleton_noisy_bool)
    skeleton = morphology.remove_small_objects(labeled, edgelength + 1)
    skeleton[skeleton > 0] = 255
    skeleton = np.uint8(skeleton)

    skeleton_corrected = remove_bug_pixels(skeleton)
    skeleton_corrected = set_black_border(skeleton_corrected)

    if plot:
        fig, axes = plt.subplots(1, 2)
        for a in axes:
            a.set_xticks([])
            a.set_yticks([])

        axes[0].imshow(thr_image, "gray")
        axes[0].set_title("thresholded")

        axes[1].imshow(skeleton_corrected, "gray")
        axes[1].set_title("skeletonised")

        plt.show()

    if save:
        cv2.imwrite(directory, skeleton_corrected )

    return np.uint8(skeleton)

def remove_bug_pixels(skeleton: np.ndarray):
    # bug pixel elimination based on
    # "Preprocessing and postprocessing for skeleton-based fingerprint minutiae extraction"
    bug_pixels = []
    for x in range(1, skeleton.shape[0] - 1):
        for y in range(1, skeleton.shape[1] - 1):
            if skeleton[x, y] == 255:
                s = num_in_4connectivity(x, y, skeleton)

                if s > 2:
                    bug_pixels.append([x, y])

    for bpx, bpy in bug_pixels:
        s = num_in_4connectivity(bpx, bpy, skeleton)

        if s > 2:
            skeleton[bpx, bpy] = 0
    return skeleton


def set_black_border(img: np.ndarray):
    mask = np.ones(img.shape, dtype=np.int8)

    mask[:, 0] = 0
    mask[:, -1] = 0
    mask[0, :] = 0
    mask[-1, :] = 0

    return np.uint8(np.multiply(mask, img))

def four_connectivity(a: int, b: int):
    # list of pixels in 4-connectivity of [a,b]
    return [[a + 1, b], [a - 1, b], [a, b + 1], [a, b - 1]]


def num_in_4connectivity(a: int, b: int, image: np.ndarray):
    # how many pixel with value 255 are in 4-connectivity of [a,b]
    neighbours = four_connectivity(a, b)

    count = 0
    for nr, nc in neighbours:
        if image[nr, nc] == 255:
            count += 1

    return count