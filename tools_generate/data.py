import os

import skimage.io as io
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tools_graph.utilz_analysis import get_data, plot_graph_on_img
from tools_generate.image import classifier_preview, classify

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def train_generator(
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    target_size,
    image_color_mode="grayscale",
    mask_color_mode="grayscale",
):
    """Image Data Generator
    Function that generates batches of data (img, mask) for training
    from specified folder. Returns images with specified pixel size
    Does preprocessing (normalization to 0-1)
    """
    # no augmentation, only rescaling
    image_datagen = ImageDataGenerator(rescale=1.0 / 255)
    mask_datagen = ImageDataGenerator(rescale=1.0 / 255)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=1,
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=1,
    )
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        mask, _ = classify(mask)
        yield (img, mask)


def get_skeletonised_ds(data_path: str, seed: int) -> tf.data.Dataset:
    skeletonised_files_glob = [
        os.path.join(data_path, "skeleton/*.png"),
        os.path.join(data_path, "**/skeleton/*.png"),
        os.path.join(data_path, "**/**/skeleton/*.png"),
    ]
    #A dataset of all files matching one or more glob patterns.
    #https://www.tensorflow.org/api_docs/python/tf/data/Dataset#list_files
    ds = tf.data.Dataset.list_files(skeletonised_files_glob, shuffle=False)

    if seed:
        #https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
        return ds.shuffle(len(ds), seed=seed, reshuffle_each_iteration=False)
    else:
        return ds

def ds_to_list(dataset: tf.data.Dataset) -> list:
    #ToDo implement the natsort function here, if the natural order is not robust!!
    return [f.decode("utf-8") for f in dataset.as_numpy_iterator()]


def get_next_filepaths_from_ds(dataset: tf.data.Dataset):
    skel_fp = next(iter(dataset)).numpy().decode("utf-8")
    graph_fp = skel_fp.replace("skeleton", "graphs").replace(".png", ".json")
    return skel_fp, graph_fp


