import sys
import os
import datetime
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
import numpy as np
from skimage import img_as_ubyte
from matplotlib.figure import Figure

SKELETONIZE_GREY_IMAGE = True
# opitimization

LOSS_PRECISION_DECIMAL_DIGITS = 1e4

LEARNING_RATE_POSE_OPT = 2*1e-4
ITERATION_POSE_OPT = 80
ALIGN_POINT_CLOUD_ON_MESH = True

SELECT_NEXT_IMAGE_BY_MIN_MATCHES = 60

DEFORMATION_RE_RENDER_TARGET_POINT_CLOUD_DURING_OPTIMIZATION = False
DEFROMATION_CHAMFER_LOSS_ELSE_EUCLIDEAN = False
TEXTURE_RENDERER_IN_BATCH_ELSE_LIST = True

WORLD_MAP_USE_TEXTURE_FOR_POSE_OPTIMIZATION = True #deprecated! not in use anymore! (is defined in task dic)

# RANSAC
RANSAC_REPROJECTION_ERROR_EPSILON = 0.6
RANSAC_ITR_POSE_OPT_PER_SAMPLE = 45
RANSAC_ITR_SAMPLES = 35
