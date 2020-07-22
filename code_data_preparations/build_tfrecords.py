import json
import cv2
import os, sys 
import numpy as np 
import skimage.transform
from glob import glob
import skimage.transform
import tensorflow as tf 
from tqdm import tqdm

MAIN_DIR = os.getcwd()
COMMON_DIR = os.path.join(MAIN_DIR, "code_commons")
sys.path.append( COMMON_DIR )
COMMON_DIR = os.path.join(MAIN_DIR, "code_training")
sys.path.append( COMMON_DIR )

import tfrecord_utils 
from tfrecord_utils import _bytes_feature

from global_constants import *  
from train_sample_generator import *

tfrecords_filename = os.path.join(MAIN_DIR, "validation.tfrecords")



if __name__ == "__main__":
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    datadir = ['/home/hikoo/project/ssd/3d_data/iphonex-recordings-umn-hcnet-dense-alignment-output/day1/groupf_video0_front_session6']
    data_generator = TrainDataGenerator( datadir, GEOMETRIC_DISTORTION = False )
    dataset = data_generator.dataset

    num_files = len(dataset)


    for i in tqdm.tqdm( range(num_files), desc=f"jpg_json to tfrecord conversion", total=num_files):
        
        jpgPath, annotations = dataset[i]

        while True:
            data_dict, validity = data_generator.generate_data( jpgPath, annotations )            
            if validity is True:
                break

        feature = {}
        data_string_dict = {}
        for name in DATA_FIELD_NAMES:
            data_string_dict[name] = data_dict[name].tostring() 
            if DATA_FIELD_TYPES is np.uint8:
                feature[name] = _bytes_feature( data_string_dict[name] )
            else:
                feature[name] = _bytes_feature( tf.compat.as_bytes( data_string_dict[name] ) )


        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

    writer.close()