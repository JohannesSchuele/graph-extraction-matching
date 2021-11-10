import numpy as np

from functions.images import apply_img_mask, threshold_imgs, skeletonise_imgs, extract_graphs

from config import Config, img_length
from video_data import video_filepath, frequency, trim_times

import warnings

warnings.simplefilter('ignore', np.RankWarning)


def after_filter(conf, skip_existing=True):
    apply_img_mask(conf)
    threshold_imgs(conf)
    skeletonise_imgs(conf)
    extract_graphs(conf, skip_existing)


if __name__ == '__main__':
    conf = Config(video_filepath, frequency,
                  img_length=img_length, trim_times=trim_times)
    after_filter(conf, skip_existing=False)
