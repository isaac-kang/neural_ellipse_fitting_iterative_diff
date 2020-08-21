import numpy as np 
import tensorflow as tf 

# IMAGE_WIDTH = 384
# IMAGE_HEIGHT = 288
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224 
IMAGE_CHANNEL = 3


QUANTIZE = False 


NUM_OF_PARAMETERS_TO_BE_ESTIMATED = (2+3) # (x,y), theta, lambda_1, lambda_2

NUM_BDRY_POINTS = 20
# DATA_FIELD_NAMES = [ "image", "mask", "mask_center", "annotated_center" ]
# DATA_FIELD_TYPES = [ np.float32, np.float32, np.float32, np.float32 ]
# DATA_FIELD_TF_TYPES = [ tf.float32, tf.float32, tf.float32, tf.float32 ]
# DATA_FIELD_SHAPES = [ (IMAGE_HEIGHT,IMAGE_WIDTH,3), (IMAGE_HEIGHT,IMAGE_WIDTH), (1,2), (1,2)]
DATA_FIELD_NAMES = ["image", "mask", "mask_dummy", "mask_center", "mask_axis_x_pts", "mask_axis_y_pts"]
DATA_FIELD_TYPES = [ np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]
DATA_FIELD_TF_TYPES = [ tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
DATA_FIELD_SHAPES = [ (IMAGE_HEIGHT,IMAGE_WIDTH,9), (IMAGE_HEIGHT,IMAGE_WIDTH), (IMAGE_HEIGHT,IMAGE_WIDTH), (1,2), (1, 4), (1, 4)]

