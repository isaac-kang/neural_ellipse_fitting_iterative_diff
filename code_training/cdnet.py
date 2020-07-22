import tensorflow as tf
import numpy as np
import os
import sys
import time
import tqdm
import graph_structures
#from modules import resfcn256

sys.path.append(os.path.join(os.getcwd(), "code_commons"))
from global_constants import *
from tfrecord_utils import *
import auxiliary_ftns

from backbone import alexnet
from train_sample_generator import *
import train_data_provider

import sharedmem
import threading

from tqdm import tqdm
from auxiliary_ftns import *

import tensorflow.contrib.slim as slim


import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope

from tensorflow.contrib import slim
from tensorflow.contrib.layers import xavier_initializer

from tensorflow.python.framework import ops

import dirt_utilities


class CDnet(object):

    def __init__(self, FLAGS=None):

        self.FLAGS = FLAGS

        # input dim
        """
        self.input_image_width = IMAGE_WIDTH
        self.input_image_height = IMAGE_HEIGHT
        self.input_num_channels = IMAGE_CHANNEL
        self.training_batch_size = self.FLAGS.batch_size

        self.input_image_size = IMAGE_WIDTH
        self.max_feature_depth = 256 # 256
        """

        triangles = []
        for i in range(NUM_BDRY_POINTS):
            triangles.append([i, (i + 1) % NUM_BDRY_POINTS, NUM_BDRY_POINTS])

        trinagles = tf.constant(triangles, dtype=tf.int32)

        self.trinagles = tf.constant(triangles, dtype=tf.int32)

    def runttime_initialize(self, add_saver=True):
        batch_slice = 1
        self.input = tf.placeholder(tf.float32, [batch_slice, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

        parameter_vectors_stf = self.add_alexnet(self.input)
        parameter_vectors_tf = self.scale_compensation(parameter_vectors_stf)

        centerx = parameter_vectors_tf[:, 0:1]
        centery = parameter_vectors_tf[:, 1:2]
        angle = parameter_vectors_tf[:, 2:3]
        radius1 = parameter_vectors_tf[:, 3:4]
        radius2 = parameter_vectors_tf[:, 4:5]
        # a > b # green
        radius1 = radius1 + radius2

        M = np.zeros(shape=[1, NUM_BDRY_POINTS + 1, 3], dtype=np.float32)
        for i in range(NUM_BDRY_POINTS):
            M[0, i, 0] = np.cos(2 * np.pi * i / NUM_BDRY_POINTS)
            M[0, i, 1] = np.sin(2 * np.pi * i / NUM_BDRY_POINTS)
            M[0, i, 2] = 1.0
        M[0, NUM_BDRY_POINTS, 0] = M[0, NUM_BDRY_POINTS, 1] = 0.0
        M[0, NUM_BDRY_POINTS, 2] = 1      # center

        M = tf.constant(M, tf.float32)
        M = tf.tile(M, [batch_slice, 1, 1])

        T = transformation(centerx, centery, angle * self.angle_scale, radius1, radius2)
        if len(T.shape) == 2:
            T = tf.expand_dims(T, axis=0)
        M = tf.matmul(M, T)

        frame_width = IMAGE_WIDTH
        frame_height = -IMAGE_HEIGHT
        cx = 0
        cy = IMAGE_HEIGHT
        output_dictionary = dirt_utilities.dirt_rendering_orthographic(M, self.trinagles, reflectances=None, frame_width=frame_width, frame_height=frame_height, cx=cx, cy=cy)
        rendering_result = output_dictionary["rendering_results"][..., 0]

        self.output = parameter_vectors_tf
        self.output_mask = rendering_result

        if QUANTIZE is True:
            tf.contrib.quantize.create_eval_graph()

        if add_saver is True:
            self.add_saver()

    def add_alexnet(self, input_tensor):
        print("add parameter estimation network")
        with tf.variable_scope("alexnet"):
            with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
                outputs, end_points = alexnet.alexnet_v2(input_tensor, NUM_OF_PARAMETERS_TO_BE_ESTIMATED, global_pool=True)
                #outputs = tf.reshape( outputs, shape = [-1, NUM_BOUNDRY_POINTS, 2])

        return outputs

    def add_validation_env(self, validation_set):
        print('initializing_tfrecord_for_validation')
        self.num_validation_samples = 0
        return

        with tf.variable_scope('tfrecord_open'):
            self.validation_set = validation_set
            validation_batch_size = 1

            num_validation_sizes = []
            num_validation_samples = 0
            for file in validation_set:
                num_samples = sum(1 for _ in tf.python_io.tf_record_iterator(file))
                num_validation_samples += num_samples
                num_validation_sizes = num_validation_sizes + [num_samples]
            print(f"validation samples = {num_validation_samples}")
            self.num_validation_samples = num_validation_samples

            self.validation_data_dict = make_batch(validation_set, validation_batch_size,
                                                   shuffle=False, num_epochs=10000, MIN_QUEUE_EXAMPLES=10)

        #self.val_input  = tf.placeholder( tf.float32, [None, IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNEL])
        #self.val_output = tf.placeholder( tf.float32, [None, IMAGE_WIDTH, IMAGE_WIDTH])

        # with tf.variable_scope('validation_stuff'):
        #    self.l2_loss = square_sum_error = tf.reduce_mean( tf.square( depth_map - self.val_output) )   #

    def evaluate_validation_loss(self, sess):
        print('validation_started')
        tic = time.time()
        validation_default_batch_size = 1
        loss_sum = 0.0

        for i in tqdm(range(self.num_validation_samples), total=self.num_validation_samples, leave=False):
            _data_dict = sess.run(self.validation_data_dict)

            _loss = sess.run(self.l2_loss, feed_dict={self.val_input: _data_dict["image"], self.val_output: _data_dict["depth"]})
            loss_sum = loss_sum + _loss

        toc = time.time()
        print(f'elapsed={toc-tic}sec')
        avg_loss = np.sqrt(loss_sum / self.num_validation_samples)
        print("avg loss = ", avg_loss)
        return avg_loss

    def scale_compensation(self, parameter_vectors_tf):
        # SCALE COMPENSATION
        # blue
        self.angle_scale = 1
        self.angle_range_scale = 1  # 0.5 mean +90, -90
        bias = [0, 0, 0, 0.5, 0.5]
        wght = [300, 300, self.angle_range_scale / self.angle_scale, 300, 200]
        offset = [IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2, 0, 0, 0]

        wght = np.array(wght, dtype=np.float32)
        wght = tf.constant(wght, dtype=tf.float32)

        bias = np.array(bias, dtype=np.float32)
        bias = tf.constant(bias, dtype=tf.float32)

        offset = np.array(offset, dtype=np.float32)
        offset = tf.constant(offset, dtype=tf.float32)

        parameter_vectors_tf = (tf.math.sigmoid(parameter_vectors_tf) - 0.5)

        scaled_parameter_vectors_tf = (parameter_vectors_tf + bias) * wght + offset
        return scaled_parameter_vectors_tf

    @staticmethod
    def normalize_scale(estimated_corners):
        s = tf.constant([IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2], dtype=tf.float32)
        s = tf.reshape(s, [1, 1, -1])

        """
        t = tf.constant( [IMAGE_HEIGHT//2,IMAGE_WIDTH//2], dtype = tf.float32 )
        t = tf.reshape( t, [1,1,-1])
        """

        estimated_corners = estimated_corners * s

        return estimated_corners

    def train_initialize(self, datadir=None, cpu_mode=False):

        # tfrecord
        if len(datadir) == 1 and os.path.splitext(datadir[0])[-1] == ".tfrecords":
            batch_data_dict = make_batch(datadir, self.FLAGS.batch_size, shuffle=True, num_epochs=10000)

        else:
            ########################## data pipeline ####################
            tic = time.time()
            # random img, mask, center in data dir -> applyt distortion
            data_generator = TrainDataGenerator(datadir)
            toc = time.time()
            print("###########################")
            print(f'data loading={toc-tic}sec')

            # generate batch
            batch_generator = train_data_provider.generate_batch(data_generator,
                                                                 batch_size=self.FLAGS.batch_size,
                                                                 num_processes=self.FLAGS.num_preprocessing_processes)

            lock = threading.Lock()

            def generate_batch():
                with lock:
                    batch_data_list = batch_generator.__next__()
                return batch_data_list
            ################################################################

            batch_list = tf.py_func(generate_batch, [], DATA_FIELD_TYPES, stateful=True)

            batch_data_dict = {}

            for idx, name in enumerate(DATA_FIELD_NAMES):
                batch_data_dict[name] = batch_list[idx]
                batch_data_dict[name].set_shape((self.FLAGS.batch_size,) + DATA_FIELD_SHAPES[idx])
                print(DATA_FIELD_NAMES[idx] + ":", batch_data_dict[name].shape, batch_data_dict[name].dtype)

        self.batch_data_dict = batch_data_dict

        if cpu_mode is True:
            gpus = ['/device:CPU:0']
            print(gpus)
        else:
            gpus = get_available_gpus()
            print(gpus)

        num_gpus = len(gpus)
        assert(self.FLAGS.batch_size % num_gpus == 0)
        batch_slice = self.FLAGS.batch_size // num_gpus

        tower_losses = []

        for idx_gpu, gpu in enumerate(gpus):
            print(gpu)
            with tf.device(gpu):
                image_slice = batch_data_dict["image"][batch_slice * idx_gpu:batch_slice * (idx_gpu + 1), ...]
                mask_slice = batch_data_dict["mask"][batch_slice * idx_gpu:batch_slice * (idx_gpu + 1), ...]
                maskcenter_slice = batch_data_dict["mask_center"][batch_slice * idx_gpu:batch_slice * (idx_gpu + 1), ...]
                mask_axis_x_pts_slice = batch_data_dict["mask_axis_x_pts"][batch_slice * idx_gpu:batch_slice * (idx_gpu + 1), ...]
                mask_axis_y_pts_slice = batch_data_dict["mask_axis_y_pts"][batch_slice * idx_gpu:batch_slice * (idx_gpu + 1), ...]

                maskcenter_slice = tf.squeeze(maskcenter_slice)
                mask_axis_x_pts_slice = tf.squeeze(mask_axis_x_pts_slice)
                mask_axis_y_pts_slice = tf.squeeze(mask_axis_y_pts_slice)

                parameter_vectors_stf = self.add_alexnet(image_slice)      # (batch, NUM_BOUNDRY_POINTS, 2) tf.float32
                parameter_vectors_tf = self.scale_compensation(parameter_vectors_stf)

                centerx = parameter_vectors_tf[:, 0:1]
                centery = parameter_vectors_tf[:, 1:2]
                angle = parameter_vectors_tf[:, 2:3]
                radius1 = parameter_vectors_tf[:, 3:4]
                radius2 = parameter_vectors_tf[:, 4:5]
                # a > b # green
                radius1 = radius1 + radius2
                self.centerx = centerx
                self.centery = centery
                self.angle = angle
                self.radius1 = radius1
                self.radius2 = radius2
                self.estimate_center = tf.concat([self.centerx, self.centery], axis=1)

                M = np.zeros(shape=[1, NUM_BDRY_POINTS + 1, 3], dtype=np.float32)
                for i in range(NUM_BDRY_POINTS):
                    M[0, i, 0] = np.cos(2 * np.pi * i / NUM_BDRY_POINTS)
                    M[0, i, 1] = np.sin(2 * np.pi * i / NUM_BDRY_POINTS)
                    M[0, i, 2] = 1.0
                M[0, NUM_BDRY_POINTS, 0] = M[0, NUM_BDRY_POINTS, 1] = 0.0
                M[0, NUM_BDRY_POINTS, 2] = 1      # center

                M = tf.constant(M, tf.float32)
                M = tf.tile(M, [batch_slice, 1, 1])

                # DRAW ELLIPSE
                T = transformation(centerx, centery, angle * self.angle_scale, radius1, radius2)
                M = tf.matmul(M, T)

                frame_width = IMAGE_WIDTH
                frame_height = -IMAGE_HEIGHT
                cx = 0
                cy = IMAGE_HEIGHT
                output_dictionary = dirt_utilities.dirt_rendering_orthographic(M, self.trinagles, reflectances=None, frame_width=frame_width, frame_height=frame_height, cx=cx, cy=cy)
                rendering_result = output_dictionary["rendering_results"][..., 0]

                error_map = tf.abs(rendering_result - mask_slice)
                error_map_raw = rendering_result - mask_slice
                region_loss = 1000.0 * tf.reduce_mean(error_map)

                center_points = M[:, -1, :2]
                area1 = tf.reduce_sum(mask_slice, axis=[1, 2])
                area2 = 3.141592 * radius1 * radius2
                area2 = tf.squeeze(area2)
                area_diff_loss = tf.reduce_mean(tf.abs(area1 - area2))
                center_distance_loss = tf.abs(center_points - maskcenter_slice)
                center_distance_loss = tf.math.maximum(center_distance_loss, 5)
                center_distance_loss = tf.reduce_sum(0.1 * center_distance_loss)
                # loss = center_distance_loss +  region_loss +  area_diff_loss # magenta
                loss = 10 * center_distance_loss + region_loss

                self.grad_centerx = tf.gradients(loss, self.centerx)
                self.grad_centery = tf.gradients(loss, self.centery)
                self.grad_angle = tf.gradients(loss, self.angle)
                self.grad_radius1 = tf.gradients(loss, self.radius1)
                self.grad_radius2 = tf.gradients(loss, self.radius2)

                self.maskcenter_slice = maskcenter_slice
                self.mask_axis_x_pts_slice = mask_axis_x_pts_slice
                self.mask_axis_y_pts_slice = mask_axis_y_pts_slice

                self.region_loss = region_loss
                with tf.variable_scope('loss'):
                    tower_losses.append(loss)

                self.add_summary_per_gpu(idx_gpu, image_slice, mask_slice, rendering_result, error_map_raw, center_points, M[:, :, :2], loss, center_distance_loss, region_loss, area_diff_loss)

        if QUANTIZE is True:
            self.add_quant_training_graph()

        """
        self.print_var_status( var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES ) )
        """

        self.loss = tf.reduce_mean(tower_losses)

        self.add_gradient()
        self.add_summary()
        self.add_saver()

    @staticmethod
    def convert_to_color_image(x):
        x = tf.expand_dims(x, axis=-1)
        x = tf.tile(x, [1, 1, 1, 3])
        return x

    @staticmethod
    def convert_raw_to_color_image(x):
        x = tf.expand_dims(x, axis=-1)
        x0 = tf.cast(tf.not_equal(x, tf.constant([0], dtype=tf.float32)), dtype=tf.float32)
        x1 = tf.cast(tf.greater(x, tf.constant([0], dtype=tf.float32)), dtype=tf.float32)
        x2 = tf.zeros_like(x)
        x = tf.concat((x0, x1, x2), axis=-1)
        return x

    def add_summary_per_gpu(self, gpu_idx, image_slice, mask_slice, rendering_result, error_map_raw, center_points, all_points, loss, center_distance_loss, region_loss, area_diff_loss):
        N = self.FLAGS.num_summary_images

        #tf.summary.histogram("estimated_corners", center_points)
        #tf.summary.histogram("gt_histogram", mask_slice)
        #tf.summary.histogram("est_histogram", rendering_result)

        with tf.variable_scope('summary_%s' % (gpu_idx)):
            tf.summary.scalar("loss_%s_th_gpu" % (gpu_idx), loss)
            tf.summary.scalar("region_loss_%s_th_gpu" % (gpu_idx), self.region_loss)
            tf.summary.scalar("center_distance_loss_%s_th_gpu" % (gpu_idx), center_distance_loss)
            # tf.summary.scalar("area_diff_loss_%s_th_gpu"%(gpu_idx), area_diff_loss )

            # 1. Input Image
            image_slice = ((image_slice) / np.sqrt(2.0) + 0.5)
            input_with_lines = tf.py_func(draw_circle, [image_slice[:N], self.maskcenter_slice[:N], (1, 0, 0), 3], tf.float32)

            # 2. Mask Image
            mask_slice = CDnet.convert_to_color_image(mask_slice)

            # 3. Estimate results on Input Image
            images_with_lines = tf.py_func(draw_contour_32f, [image_slice[:N], all_points[:N]], tf.float32)
            images_with_lines = tf.py_func(draw_circle, [images_with_lines[:N], self.maskcenter_slice[:N], (1, 0, 0), 3], tf.float32)
            images_with_lines = tf.py_func(draw_circle, [images_with_lines[:N], self.estimate_center[:N], (1, 1, 0), 2], tf.float32)
            images_with_lines = tf.py_func(draw_angle, [images_with_lines[:N], self.estimate_center[:N], self.radius1[:N], self.angle[:N], self.angle_scale,
                                                        self.grad_angle[:N], self.mask_axis_x_pts_slice, self.mask_axis_y_pts_slice], tf.float32)

            # 4. Estimate rener
            rendering_result = CDnet.convert_to_color_image(rendering_result)
            rendering_result = tf.py_func(draw_grad, [rendering_result[:N], self.grad_centerx[:N], self.grad_centery[:N], self.grad_angle[:N],
                                                      self.grad_radius1[:N], self.grad_radius2[:N]], tf.float32)

            # 5. Error map
            error_map = CDnet.convert_raw_to_color_image(error_map_raw)

            """
        
            images_with_lines = tf.py_func( draw_triangles, \
                [ image_slice[:N], estimated_corners[:N] ], tf.float32 )            
            """

            input_prediction = 255.0 * tf.concat(
                [input_with_lines[:N],
                 mask_slice[:N],
                 images_with_lines,
                 rendering_result[:N],
                 error_map[:N]
                 ], axis=2)
            input_prediction = tf.clip_by_value(input_prediction, 0.0, 255.0)

            input_prediction = tf.image.resize_bilinear(input_prediction, (IMAGE_HEIGHT, IMAGE_WIDTH * 5))
            tf.summary.image("input_prediction gpu:%s" % (gpu_idx), input_prediction, max_outputs=N)

    def add_saver(self):
        self.saver = tf.train.Saver(max_to_keep=5)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.regression_saver = tf.train.Saver(var_list=var_list)

    def add_quant_training_graph(self):
        # Call the training rewrite which rewrites the graph in-place with
        # FakeQuantization nodes and folds batchnorm for training. It is
        # often needed to fine tune a floating point model for quantization
        # with this training tool. When training from scratch, quant_delay
        # can be used to activate quantization after training to converge
        # with the float graph, effectively fine-tuning the model.
        print("add quants")
        g = tf.get_default_graph()
        tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=2000)

    def print_var_status(self, var_list):
        print("Following tensors will be updated:")
        sum_tensor_size = 0
        for v in var_list:
            cur_tensor_size = auxiliary_ftns.tensor_size(v)
            print(f"{v.name} with the size of {cur_tensor_size}")
            sum_tensor_size += cur_tensor_size
        print(f"total size = {sum_tensor_size} ({sum_tensor_size})")

    def add_gradient(self):
        print("add gradient")
        with tf.variable_scope("gradients"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            self.global_step = tf.Variable(0, trainable=False)

            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.FLAGS.learning_rate,
                global_step=self.global_step,
                decay_steps=self.FLAGS.num_samples_per_learning_rate_half_decay / self.FLAGS.batch_size,
                decay_rate=0.5)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)

            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def add_summary(self):
        with tf.variable_scope('summary'):
            tf.summary.scalar("loss", self.loss)
            self.summaries = tf.summary.merge_all()

    def occasional_jobs(self, sess, global_step):
        ckpt_filename = os.path.join(self.FLAGS.train_dir, "myckpt")

        if global_step % self.FLAGS.save_every == 0:
            save_path = self.saver.save(sess, ckpt_filename, global_step=global_step)
            tqdm.write("saved at" + save_path)

        # if global_step % self.FLAGS.eval_every == 0 and self.num_validation_samples != 0:
        #     eval_loss = self.evaluate_validation_loss(sess)
        #     print("evaluation loss:", eval_loss)
        #     summary = tf.Summary()
        #     with tf.variable_scope("validation"):
        #         summary.value.add(tag="validation_loss", simple_value=eval_loss)
        #         self.writer.add_summary(summary, global_step=global_step)

        # write examples to the examples directory
        """
        if  global_step % self.FLAGS.save_examples_every == 0:
            print("save examples - nothing done")            
        """

    def train(self, sess):
        self.writer = tf.summary.FileWriter(self.FLAGS.train_dir, sess.graph)

        exp_loss = None
        counter = 0

        print("train starting")

        print_every = 1000
        while True:
            for iter in tqdm(range(print_every), leave=False):

                output_feed = {
                    "train_op": self.train_op,
                    "global_step": self.global_step,
                    "learning_rate": self.learning_rate,
                    "loss": self.loss
                }

                if iter % self.FLAGS.summary_every == 0:
                    output_feed["summaries"] = self.summaries

                _results = sess.run(output_feed)

                # g_centerx = np.array(sess.run(self.grad_centerx))
                # g_centery = np.array(sess.run(self.grad_centery))
                g_angle = np.array(sess.run(self.grad_angle))
                # g_radius1 = np.array(sess.run(self.grad_radius1))
                # g_radius2 = np.array(sess.run(self.grad_radius2))
                # g_par = np.concatenate((g_centerx, g_centery, g_angle, g_radius1, g_radius2), axis=0)
                # print(g_par)

                global_step = _results["global_step"]
                learning_rate = _results["learning_rate"]

                if iter % self.FLAGS.summary_every == 0:
                    self.writer.add_summary(_results["summaries"], global_step=global_step)

                cur_loss = _results["loss"]

                if not exp_loss:  # first iter
                    exp_loss = cur_loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * cur_loss

                self.occasional_jobs(sess, global_step)

            if True:  # global_step  % print_every == 0:
                print(f"global_step = {global_step}, learning_rate = {learning_rate:.6f}")
                print(f"loss = {exp_loss:0.4f}")

        sys.stdout.flush()


if __name__ == "__main__":
    cdnet = CDnet()

    frame_width = IMAGE_WIDTH
    frame_height = -IMAGE_HEIGHT
    cx = 0
    cy = IMAGE_HEIGHT

    parameter_vectors = np.random.randint(0, IMAGE_WIDTH, size=[1, NUM_BOUNDRY_POINTS + 1, 2])
    parameter_vectors[0, 0, :] = np.array([0, 0], dtype=np.float32)
    parameter_vectors[0, 4, :] = np.array([IMAGE_WIDTH, IMAGE_HEIGHT], dtype=np.float32)
    parameter_vectors[0, 8, :] = np.array([IMAGE_WIDTH, 0], dtype=np.float32)

    parameter_vectors_tf = tf.constant(parameter_vectors, dtype=tf.float32)

    output_dictionary = dirt_utilities.dirt_rendering(parameter_vectors_tf, cdnet.trinagles, reflectances=None, frame_width=frame_width, frame_height=frame_height, cx=cx, cy=cy)

    with tf.Session() as sess:
        _output_dictionary = sess.run(output_dictionary)

    cv2.imshow("wnd", _output_dictionary["rendering_results"][0, ..., 0])
    cv2.waitKey(0)
