import tensorflow as tf
import numpy as np
import os
import sys
import time
import graph_structures
import cv2
import sharedmem
import threading
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import tensorflow.contrib.slim as slim
import dirt_utilities
import tqdm
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import slim
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.framework import ops
sys.path.append(os.path.join(os.getcwd(), "code_commons"))
from global_constants import *
import auxiliary_ftns
from auxiliary_ftns import *
from backbone import alexnet
from train_sample_generator import *
import train_data_provider
from tqdm import tqdm

class CDnet(object):

    def __init__(self, args=None):
        self.args = args
        triangles = []
        for i in range(NUM_BDRY_POINTS):
            triangles.append([i, (i + 1) % NUM_BDRY_POINTS, NUM_BDRY_POINTS])
        trinagles = tf.constant(triangles, dtype=tf.int32)
        self.trinagles = tf.constant(triangles, dtype=tf.int32)

    def add_alexnet(self, input_tensor):
        print("add parameter estimation network")
        with tf.variable_scope("alexnet"):
            with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
                outputs, end_points = alexnet.alexnet_v2(input_tensor, NUM_OF_PARAMETERS_TO_BE_ESTIMATED, global_pool=True)
        return outputs

    def scale_compensation(self, parameter_vectors_tf):
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

    def make_batch_data_dict(self, datadir):
        tic = time.time()
        data_generator = TrainDataGenerator(datadir)
        toc = time.time()
        print("---------------------------")
        print(f'data loading={toc-tic}sec')
        print("---------------------------")
        batch_generator = train_data_provider.generate_batch(data_generator, batch_size=self.args.batch_size, num_processes=self.args.num_preprocessing_processes)
        lock = threading.Lock()
        def generate_batch():
            with lock:
                batch_data_list = batch_generator.__next__()
            return batch_data_list
        batch_list = tf.py_func(generate_batch, [], DATA_FIELD_TYPES, stateful=True)
        batch_data_dict = {}
        for idx, name in enumerate(DATA_FIELD_NAMES):
            batch_data_dict[name] = batch_list[idx]
            batch_data_dict[name].set_shape((self.args.batch_size,) + DATA_FIELD_SHAPES[idx])
            print(DATA_FIELD_NAMES[idx] + ":", batch_data_dict[name].shape, batch_data_dict[name].dtype)
        return batch_data_dict

    def render_ellipse(self, batch_slice, centerx, centery, angle, radius1, radius2):
        M = np.zeros(shape=[1, NUM_BDRY_POINTS + 1, 3], dtype=np.float32)
        for i in range(NUM_BDRY_POINTS):
            M[0, i, 0] = np.cos(2 * np.pi * i / NUM_BDRY_POINTS)
            M[0, i, 1] = np.sin(2 * np.pi * i / NUM_BDRY_POINTS)
            M[0, i, 2] = 1.0
        M[0, NUM_BDRY_POINTS, 0] = M[0, NUM_BDRY_POINTS, 1] = 0.0
        M[0, NUM_BDRY_POINTS, 2] = 1
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
        return rendering_result, M

    def model(self, input_1, batch_slice=1):
        #========= STAGE 1 =============
        parameter_vectors_stf_1 = self.add_alexnet(input_1)
        parameter_vectors_tf_1 = self.scale_compensation(parameter_vectors_stf_1)

        centerx_1 = parameter_vectors_tf_1[:, 0:1]
        centery_1 = parameter_vectors_tf_1[:, 1:2]
        angle_1 = parameter_vectors_tf_1[:, 2:3]
        radius1_1 = parameter_vectors_tf_1[:, 3:4]
        radius2_1 = parameter_vectors_tf_1[:, 4:5]
        # a > b
        radius1_1 = radius1_1 + radius2_1
        self.centerx_1 = centerx_1
        self.centery_1 = centery_1
        self.angle_1 = angle_1
        self.radius1_1 = radius1_1
        self.radius2_1 = radius2_1
        self.estimate_center_1 = tf.concat([self.centerx_1, self.centery_1], axis=1)

        # DRAW ELLIPSE (5 estimated parameters -> 20 points + center -> triangularization -> rendering)
        rendering_result_1, self.M_1 = self.render_ellipse(batch_slice, centerx_1, centery_1, angle_1, radius1_1, radius2_1)

        #========= STAGE 2 =============
        input_2 = tf.concat([input_1[:, :, :, :4], tf.expand_dims(rendering_result_1, axis=-1)], axis=-1)
        parameter_vectors_stf_2 = self.add_alexnet(input_2)
        parameter_vectors_tf_2 = self.scale_compensation(parameter_vectors_stf_2)

        centerx_2 = parameter_vectors_tf_2[:, 0:1]
        centery_2 = parameter_vectors_tf_2[:, 1:2]
        angle_2 = parameter_vectors_tf_2[:, 2:3]
        radius1_2 = parameter_vectors_tf_2[:, 3:4]
        radius2_2 = parameter_vectors_tf_2[:, 4:5]
        # a > b
        radius1_2 = radius1_2 + radius2_2
        self.centerx_2 = centerx_2
        self.centery_2 = centery_2
        self.angle_2 = angle_2
        self.radius1_2 = radius1_2
        self.radius2_2 = radius2_2
        self.estimate_center_2 = tf.concat([self.centerx_2, self.centery_2], axis=1)

        # DRAW ELLIPSE (5 estimated parameters -> 20 points + center -> triangularization -> rendering)
        rendering_result_2, self.M_2 = self.render_ellipse(batch_slice, centerx_2, centery_2, angle_2, radius1_2, radius2_2)
        
        #========= STAGE 3 =============
        input_3 = tf.concat([input_1[:, :, :, :4], tf.expand_dims(rendering_result_2, axis=-1)], axis=-1)
        parameter_vectors_stf_3 = self.add_alexnet(input_3)
        parameter_vectors_tf_3 = self.scale_compensation(parameter_vectors_stf_3)

        centerx_3 = parameter_vectors_tf_3[:, 0:1]
        centery_3 = parameter_vectors_tf_3[:, 1:2]
        angle_3 = parameter_vectors_tf_3[:, 2:3]
        radius1_3 = parameter_vectors_tf_3[:, 3:4]
        radius2_3 = parameter_vectors_tf_3[:, 4:5]
        # a > b
        radius1_3 = radius1_3 + radius2_3
        self.centerx_3 = centerx_3
        self.centery_3 = centery_3
        self.angle_3 = angle_3
        self.radius1_3 = radius1_3
        self.radius2_3 = radius2_3
        self.estimate_center_3 = tf.concat([self.centerx_3, self.centery_3], axis=1)

        # DRAW ELLIPSE (5 estimated parameters -> 20 points + center -> triangularization -> rendering)
        rendering_result_3, self.M_3 = self.render_ellipse(batch_slice, centerx_3, centery_3, angle_3, radius1_3, radius2_3)
        
        #========= STAGE 2 =============
        input_4 = tf.concat([input_1[:, :, :, :4], tf.expand_dims(rendering_result_3, axis=-1)], axis=-1)
        parameter_vectors_stf_4 = self.add_alexnet(input_4)
        parameter_vectors_tf_4 = self.scale_compensation(parameter_vectors_stf_4)

        centerx_4 = parameter_vectors_tf_4[:, 0:1]
        centery_4 = parameter_vectors_tf_4[:, 1:2]
        angle_4 = parameter_vectors_tf_4[:, 2:3]
        radius1_4 = parameter_vectors_tf_4[:, 3:4]
        radius2_4 = parameter_vectors_tf_4[:, 4:5]
        # a > b
        radius1_4 = radius1_4 + radius2_4
        self.centerx_4 = centerx_4
        self.centery_4 = centery_4
        self.angle_4 = angle_4
        self.radius1_4 = radius1_4
        self.radius2_4 = radius2_4
        self.estimate_center_4 = tf.concat([self.centerx_4, self.centery_4], axis=1)

        # DRAW ELLIPSE (5 estimated parameters -> 20 points + center -> triangularization -> rendering)
        rendering_result_4, self.M_4 = self.render_ellipse(batch_slice, centerx_4, centery_4, angle_4, radius1_4, radius2_4)
        
        return parameter_vectors_tf_4, rendering_result_4

    def runttime_initialize(self, add_saver=True):
        batch_slice = 1
        self.input = tf.placeholder(tf.float32, [batch_slice, IMAGE_HEIGHT, IMAGE_WIDTH, 5])
        parameter_vectors_tf, rendering_result = self.model(self.input)
        self.output = parameter_vectors_tf
        self.output_mask = rendering_result
        if add_saver is True:
            self.add_saver()

    def train_initialize(self, datadir=None, cpu_mode=False):
        # make batch
        batch_data_dict = self.make_batch_data_dict(datadir)
        # assign gpu
        gpus = get_available_gpus()
        num_gpus = len(gpus)
        assert(self.args.batch_size % num_gpus == 0)
        batch_slice = self.args.batch_size // num_gpus
        tower_losses = []
        print('gpus : ', gpus, self.args.gpu)
        gpus = [gpus[int(self.args.gpu)]]
        for idx_gpu, gpu in enumerate(gpus):
            print(gpu)
            with tf.device(gpu):
                # load batch
                image_slice = batch_data_dict["image"][batch_slice * idx_gpu:batch_slice * (idx_gpu + 1), ...]
                mask_slice = batch_data_dict["mask"][batch_slice * idx_gpu:batch_slice * (idx_gpu + 1), ...]
                mask_dummy_slice = batch_data_dict["mask_dummy"][batch_slice * idx_gpu:batch_slice * (idx_gpu + 1), ...]
                maskcenter_slice = batch_data_dict["mask_center"][batch_slice * idx_gpu:batch_slice * (idx_gpu + 1), ...]
                mask_axis_x_pts_slice = batch_data_dict["mask_axis_x_pts"][batch_slice * idx_gpu:batch_slice * (idx_gpu + 1), ...]
                mask_axis_y_pts_slice = batch_data_dict["mask_axis_y_pts"][batch_slice * idx_gpu:batch_slice * (idx_gpu + 1), ...]
                maskcenter_slice = tf.squeeze(maskcenter_slice)
                mask_axis_x_pts_slice = tf.squeeze(mask_axis_x_pts_slice)
                mask_axis_y_pts_slice = tf.squeeze(mask_axis_y_pts_slice)
                mask_dummy_slice = tf.expand_dims(mask_dummy_slice, axis=-1)
                self.maskcenter_slice = maskcenter_slice
                self.mask_axis_x_pts_slice = mask_axis_x_pts_slice
                self.mask_axis_y_pts_slice = mask_axis_y_pts_slice
                
                input_1 = tf.concat([image_slice, mask_dummy_slice], axis=-1)
                parameter_vectors_tf_2, rendering_result_2 = self.model(input_1, batch_slice=batch_slice)

                #======= LOSS =================

                # region loss
                error_map = tf.abs(rendering_result_2 - mask_slice)
                error_map_raw = rendering_result_2 - mask_slice
                region_loss_gpu = 1000.0 * tf.reduce_mean(error_map)

                # center point loss
                center_points = self.M_4[:, -1, :2]
                center_distance_loss = tf.abs(center_points - maskcenter_slice)
                center_distance_loss = tf.math.maximum(center_distance_loss, 5)
                center_distance_loss_gpu = tf.reduce_sum(0.1 * center_distance_loss)
                
                loss_gpu = 10 * center_distance_loss_gpu + region_loss_gpu

                self.grad_angle_1 = tf.gradients(loss_gpu, self.angle_1)
                self.grad_angle_2 = tf.gradients(loss_gpu, self.angle_2)
                self.grad_angle_3 = tf.gradients(loss_gpu, self.angle_3)
                self.grad_angle_4 = tf.gradients(loss_gpu, self.angle_4)

                with tf.variable_scope('loss'):
                    tower_losses.append(loss_gpu)

                self.add_summary_per_gpu(idx_gpu, image_slice, error_map_raw,
                    region_loss_gpu, center_distance_loss_gpu, loss_gpu,
                    self.M_1[:, :, :2], self.M_2[:, :, :2], self.M_3[:, :, :2], self.M_4[:, :, :2],
                    )

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

    def add_summary_per_gpu(self, gpu_idx, image_slice, error_map_raw, region_loss, center_distance_loss, loss,
        M_1, M_2, M_3, M_4):
        N = self.args.num_summary_images

        with tf.variable_scope('summary_%s' % (gpu_idx)):
            tf.summary.scalar("loss_%s_th_gpu" % (gpu_idx), loss)
            tf.summary.scalar("region_loss_%s_th_gpu" % (gpu_idx), region_loss)
            tf.summary.scalar("center_distance_loss_%s_th_gpu" % (gpu_idx), center_distance_loss)

            # # Edge map 
            # edge_slice = ((image_slice) / np.sqrt(2.0) + 0.5)[:, :, :, 3]
            # edge_slice = CDnet.convert_to_color_image(edge_slice)

            # Input Image
            image_slice = ((image_slice) / np.sqrt(2.0) + 0.5)[:, :, :, :3]
            input_with_lines = tf.py_func(draw_circle, [image_slice[:N], self.maskcenter_slice[:N], (1, 0, 0), 3], tf.float32)

            # STAGE1 estimate results on Input Image
            images_with_lines_1 = tf.py_func(draw_contour_32f, [image_slice[:N], M_1[:, :, :2][:N]], tf.float32)
            images_with_lines_1 = tf.py_func(draw_circle, [images_with_lines_1[:N], self.maskcenter_slice[:N], (1, 0, 0), 3], tf.float32)
            images_with_lines_1 = tf.py_func(draw_circle, [images_with_lines_1[:N], self.estimate_center_1[:N], (1, 1, 0), 2], tf.float32)
            images_with_lines_1 = tf.py_func(draw_angle, [images_with_lines_1[:N], self.estimate_center_1[:N], self.radius1_1[:N], self.angle_1[:N], self.angle_scale,
                                                        self.grad_angle_1[:N], self.mask_axis_x_pts_slice, self.mask_axis_y_pts_slice], tf.float32)

            # STAGE2 estimate results on Input Image
            images_with_lines_2 = tf.py_func(draw_contour_32f, [image_slice[:N], M_2[:, :, :2][:N]], tf.float32)
            images_with_lines_2 = tf.py_func(draw_circle, [images_with_lines_2[:N], self.maskcenter_slice[:N], (1, 0, 0), 3], tf.float32)
            images_with_lines_2 = tf.py_func(draw_circle, [images_with_lines_2[:N], self.estimate_center_2[:N], (1, 1, 0), 2], tf.float32)
            images_with_lines_2 = tf.py_func(draw_angle, [images_with_lines_2[:N], self.estimate_center_2[:N], self.radius1_2[:N], self.angle_2[:N], self.angle_scale,
                                                        self.grad_angle_2[:N], self.mask_axis_x_pts_slice, self.mask_axis_y_pts_slice], tf.float32)

            # STAGE3 estimate results on Input Image
            images_with_lines_3 = tf.py_func(draw_contour_32f, [image_slice[:N], M_3[:, :, :2][:N]], tf.float32)
            images_with_lines_3 = tf.py_func(draw_circle, [images_with_lines_3[:N], self.maskcenter_slice[:N], (1, 0, 0), 3], tf.float32)
            images_with_lines_3 = tf.py_func(draw_circle, [images_with_lines_3[:N], self.estimate_center_3[:N], (1, 1, 0), 2], tf.float32)
            images_with_lines_3 = tf.py_func(draw_angle, [images_with_lines_3[:N], self.estimate_center_3[:N], self.radius1_3[:N], self.angle_3[:N], self.angle_scale,
                                                        self.grad_angle_3[:N], self.mask_axis_x_pts_slice, self.mask_axis_y_pts_slice], tf.float32)

            # STAGE2 estimate results on Input Image
            images_with_lines_4 = tf.py_func(draw_contour_32f, [image_slice[:N], M_4[:, :, :2][:N]], tf.float32)
            images_with_lines_4 = tf.py_func(draw_circle, [images_with_lines_4[:N], self.maskcenter_slice[:N], (1, 0, 0), 3], tf.float32)
            images_with_lines_4 = tf.py_func(draw_circle, [images_with_lines_4[:N], self.estimate_center_4[:N], (1, 1, 0), 2], tf.float32)
            images_with_lines_4 = tf.py_func(draw_angle, [images_with_lines_4[:N], self.estimate_center_4[:N], self.radius1_4[:N], self.angle_4[:N], self.angle_scale,
                                                        self.grad_angle_4[:N], self.mask_axis_x_pts_slice, self.mask_axis_y_pts_slice], tf.float32)

            # Error map
            error_map = CDnet.convert_raw_to_color_image(error_map_raw)

            # Combine together
            input_prediction = 255.0 * tf.concat(
                [input_with_lines[:N],
                images_with_lines_1[:N],
                images_with_lines_2[:N],
                images_with_lines_3[:N],
                images_with_lines_4[:N],
                error_map[:N]
                 ], axis=2)
            input_prediction = tf.clip_by_value(input_prediction, 0.0, 255.0)

            self.input_prediction = tf.image.resize_bilinear(input_prediction, (IMAGE_HEIGHT, IMAGE_WIDTH * 5))
            tf.summary.image("input_prediction gpu:%s" % (gpu_idx), self.input_prediction, max_outputs=N)


    def add_saver(self):
        self.saver = tf.train.Saver(max_to_keep=5)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.regression_saver = tf.train.Saver(var_list=var_list)

    def add_gradient(self):
        print("add gradient")
        with tf.variable_scope("gradients"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            self.global_step = tf.Variable(0, trainable=False)

            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.args.learning_rate,
                global_step=self.global_step,
                decay_steps=self.args.num_samples_per_learning_rate_half_decay / self.args.batch_size,
                decay_rate=0.5)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)

            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def add_summary(self):
        with tf.variable_scope('summary'):
            tf.summary.scalar("loss", self.loss)
            self.summaries = tf.summary.merge_all()

    def occasional_jobs(self, sess, global_step):
        ckpt_filename = os.path.join(self.args.train_dir, "myckpt")

        if global_step % self.args.save_every == 0:
            save_path = self.saver.save(sess, ckpt_filename, global_step=global_step)
            tqdm.write("saved at" + save_path)

    def train(self, sess):
        self.writer = tf.summary.FileWriter(self.args.train_dir, sess.graph)

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

                if iter % self.args.summary_every == 0:
                    output_feed["summaries"] = self.summaries
                    output_feed["save_images"] = self.input_prediction

                _results = sess.run(output_feed)

                global_step = _results["global_step"]
                learning_rate = _results["learning_rate"]

                if iter % self.args.summary_every == 0:
                    self.writer.add_summary(_results["summaries"], global_step=global_step)
                    save_images = _results["save_images"]
                    for idx, input_pred in enumerate(save_images):
                        save_img = input_pred if idx == 0 else np.concatenate((save_img, input_pred), axis=0)
                    save_img = cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB)
                    number = np.random.randint(100)
                    if not os.path.exists(self.args.save_imgs_dir):
                        os.makedirs(self.args.save_imgs_dir)
                    cv2.imwrite(self.args.save_imgs_dir + '/' + str(number).zfill(2) +'.png', save_img)

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
    pass