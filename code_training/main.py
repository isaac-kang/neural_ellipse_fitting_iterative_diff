import numpy as np
import tensorflow as tf
import os
import sys
import cv2

sys.path.append(os.path.join(os.getcwd(), "code_commons"))
import set_default_training_options
import training_help_ftns
from cdnet import CDnet

from global_constants import *
from auxiliary_ftns import *
from model_test import *


MAIN_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(MAIN_DIR, "../experiments")
FLAGS = set_default_training_options.get_flags(MAIN_DIR)

def main(args):
    #==================
    #     Config
    #==================
    if sys.version_info[0] != 3:
        raise Exception(f"ERROR: You must use Python 3.7 "
                        f" but you are running Python {sys.version_info[0]}")
    print(f"This code was developed and tested on TensorFlow 1.13.0. " f"Your TensorFlow version: {tf.__version__}.")
    if not FLAGS.experiment_name:
        raise Exception("You need to specify an --experiment_name or --train_dir.")
    FLAGS.experiment_name = experiment_name = FLAGS.experiment_name
    FLAGS.train_dir = train_dir = (FLAGS.train_dir or os.path.join(OUTPUT_DIR, experiment_name))
    if not os.path.exists(train_dir):
        ensure_dir(train_dir)

    print("###########################")
    print(f"experiment_name: {experiment_name}")
    print(f"train_dir: {train_dir}")
    print(f"batch size: {FLAGS.batch_size}")
    print(f"mode: {FLAGS.mode}")
    print(f"data_dir: {FLAGS.data_dir}")
    print(f"validation tfrecord: {FLAGS.validation_dataset_file_path}")
    print(f"num_samples_per_learning_rate_half_decay: {FLAGS.num_samples_per_learning_rate_half_decay}")
    print("###########################")


    cdnet = CDnet(FLAGS)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction=1.0
    config.allow_soft_placement = True
    config.log_device_placement = False

    if FLAGS.mode == "train":
        os.system(f'rm {train_dir}/events*')
        cdnet.train_initialize(FLAGS.data_dir, cpu_mode=False)
        with tf.Session(config=config) as sess:
            training_help_ftns.initialize_model(sess, cdnet, train_dir)
            cdnet.train(sess)

    elif FLAGS.mode == 'test':
        cdnet.runttime_initialize()
        with tf.Session(config=config) as sess:
            ckpt_basename = training_help_ftns.initialize_model(sess, cdnet, train_dir, expect_exists=True)
            test_path = '../data/test/' + FLAGS.test_dir + '/*'
            print('\n==========\n  TEST   \n==========')
            print('ckpt : ', ckpt_basename[0])
            print('test_path : ', test_path)
            run_test(sess, cdnet, FLAGS, 'folder', frame_size=(IMAGE_WIDTH, IMAGE_HEIGHT), srcname=test_path, save_video=False, ckpt_basename=ckpt_basename[1])
            os._exit(0)
    os._exit(0)

if __name__ == '__main__':
    tf.app.run()
