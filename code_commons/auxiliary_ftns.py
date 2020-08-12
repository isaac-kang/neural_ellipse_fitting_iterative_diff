import functools
import cv2
import numpy as np
from global_constants import *
from tensorflow.python.framework import ops
import os

drawthickline = True

import re


def transformation(cx, cy, theta, lambda1, lambda2, name=None):
    # DRAW ELLIPSE

    # x is indexed by *, x/y/z
    with ops.name_scope(name, 'transformation', []) as scope:
        #x = tf.convert_to_tensor(cx, name='x')
        zeros = tf.zeros_like(cx)  # indexed by *
        ones = tf.ones_like(zeros)

        elements = [
            [lambda1 * tf.cos(2 * np.pi * theta), lambda2 * tf.sin(2 * np.pi * theta), cx],
            [-lambda1 * tf.sin(2 * np.pi * theta), lambda2 * tf.cos(2 * np.pi * theta), cy],
            [zeros, zeros, ones]]

        return tf.squeeze(tf.transpose(tf.convert_to_tensor(elements, dtype=tf.float32)))

def draw_angle(img, center, radius, angle, angle_scale, grad_angle, r1_pts, r2_pts):
    img = img.copy()
    for i in range(img.shape[0]):
        angle[i, 0] = angle[i, 0] * angle_scale
        start_pt = (center[i, 0], center[i, 1])
        end_pt = (start_pt[0] + radius[i] * np.cos(2 * np.pi * angle[i])), start_pt[1] - radius[i] * np.sin(2 * np.pi * angle[i])
        sign = 1 if grad_angle[0, i, 0] < 0 else -1
        l = 30
        grad_pt = (end_pt[0] + l * np.cos(2 * np.pi * angle[i] + sign * 0.5 * np.pi), end_pt[1] - l * np.sin(2 * np.pi * angle[i] + sign * 0.5 * np.pi))
        # current angle
        cv2.line(img[i], start_pt, end_pt, color=(1, 1, 0), thickness=1)
        # angle gradient direction
        cv2.line(img[i], end_pt, grad_pt, color=(0, 0, 1), thickness=2)
        # axis_x
        cv2.line(img[i], (r1_pts[i, 0], r1_pts[i, 1]), (r1_pts[i, 2], r1_pts[i, 3]), (1, 0, 0), thickness=1)
        # axis_y
        cv2.line(img[i], (r2_pts[i, 0], r2_pts[i, 1]), (r2_pts[i, 2], r2_pts[i, 3]), (1, 0, 1), thickness=1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_angle = angle[i, 0] * 360
        # cv2.putText(img[i], str(text_angle), (10, 10), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        # text_grad_angle = grad_angle[0, i, 0]
        # cv2.putText(img[i], str(text_grad_angle),  (10,30), font, 0.4, (0,0,0), 2, cv2.LINE_AA)
    return img


def is_number_regex(s):
    """ Returns True is string is a number. """
    if re.match("^\d+?\.\d+?$", s) is None:
        return s.isdigit()
    return True


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    result = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if len(result) > 0:
        return result
    return [x.name for x in local_device_protos]


def tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()), 1)


def draw_points(img, landmarks, color=(0, 0, 255)):

    img = img.copy()
    num_images = landmarks.shape[0]
    num_landmarks = landmarks.shape[1]
    assert num_images == img.shape[0]

    for img_idx in range(num_images):

        for j in range(num_landmarks):
            x0 = landmarks[img_idx, j, 0]
            y0 = landmarks[img_idx, j, 1]
            cv2.circle(img[img_idx, ...], (x0, y0), 2, color, 3)
    return img


def bounded_min_max(maps, min_value, max_value):
    minc = np.zeros(shape=[maps.shape[0], 1, 1])
    maxc = np.zeros(shape=[maps.shape[0], 1, 1])

    for i in range(maps.shape[0]):
        curmap = maps[i]
        x = curmap[np.where(curmap > min_value)]
        minc[i, 0, 0] = np.min(x)
        x = curmap[np.where(curmap < max_value)]
        maxc[i, 0, 0] = np.max(x)

    return minc, maxc


def depth_normalize(_depth_maps, mindist, maxdist, out_channel=3):

    minc, maxc = bounded_min_max(_depth_maps, mindist, maxdist)

    #maxc = np.max( _depth_maps, axis=(1,2)).reshape(-1,1,1)
    #minc = np.min( _depth_maps, axis=(1,2)).reshape(-1,1,1)
    _depth_maps = (_depth_maps - minc) / (maxc - minc)
    _depth_maps = np.clip(_depth_maps, 0, 1)

    if len(_depth_maps.shape) == 3 and out_channel == 3:
        _depth_maps = np.expand_dims(_depth_maps, axis=-1)
        _depth_maps = np.tile(_depth_maps, [1, 1, 1, 3])
        _depth_maps[np.where(_depth_maps >= 1.0)] = 0.0

    return _depth_maps


def draw_contour_32f(img, landmarks):
    img = img.copy()
    for i in range(img.shape[0]):
        pts = landmarks[i].copy()
        #pts[:,0] = pts[:,0] + IMAGE_WIDTH//2
        #pts[:,1] = - pts[:,1] + IMAGE_HEIGHT//2

        draw_contour(img[i], pts, 0, NUM_BDRY_POINTS - 1, c=(1, 1, 0))
    return img


def draw_contour(img, landmarks, start_idx, end_idx, c):
    thickness = 1  # np.random.randint(0,3) % 2+ 1
    if drawthickline is True:
        thickness = 2
    for j in range(start_idx, end_idx):

        pt1 = (landmarks[j, 0], landmarks[j, 1])
        pt2 = (landmarks[(j + 1), 0], landmarks[(j + 1), 1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, str(j),  pt1, font, .2, (0,255,0), 2, cv2.LINE_AA)
        cv2.line(img, pt1, pt2, color=c, thickness=thickness)

    pt1 = (landmarks[start_idx, 0], landmarks[start_idx, 1])
    pt2 = (landmarks[end_idx, 0], landmarks[end_idx, 1])
    cv2.line(img, pt1, pt2, color=c, thickness=thickness)


def draw_circle(img, center, color, size):
    img = img.copy()
    color = tuple([int(x) for x in color])
    for i in range(img.shape[0]):
        pt = (center[i, 0], center[i, 1])
        cv2.circle(img[i], pt, size, color, thickness=-1)
    return img


def draw_grad(img, grad_centerx, grad_centery, grad_angle, grad_radius1, grad_radius2):
    img = img.copy()
    for i in range(img.shape[0]):
        g1 = -grad_centerx[0, i, 0]
        g2 = grad_centery[0, i, 0]
        g3 = -grad_angle[0, i, 0]
        g4 = -grad_radius1[0, i, 0]
        g5 = -grad_radius2[0, i, 0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        if True:
            g1 = '+' if np.sign(g1) == 1.0 else '-'
            g2 = '+' if np.sign(g2) == 1.0 else '-'
            g3 = '+' if np.sign(g3) == 1.0 else '-'
            g4 = '+' if np.sign(g4) == 1.0 else '-'
            g5 = '+' if np.sign(g5) == 1.0 else '-'
        fontscale = 0.5
        thickness = 1
        cv2.putText(img[i], str(g1), (10, 10), font, fontscale, (0, 1, 1), thickness, cv2.LINE_AA)
        cv2.putText(img[i], str(g2), (10, 30), font, fontscale, (0, 1, 1), thickness, cv2.LINE_AA)
        cv2.putText(img[i], str(g3), (10, 50), font, fontscale, (0, 1, 1), thickness, cv2.LINE_AA)
        cv2.putText(img[i], str(g4), (10, 70), font, fontscale, (0, 1, 1), thickness, cv2.LINE_AA)
        cv2.putText(img[i], str(g5), (10, 90), font, fontscale, (0, 1, 1), thickness, cv2.LINE_AA)
    return img


def draw_contour_list(img, landmarks, countour_list, c, thickness=1):

    if drawthickline is True:
        thickness = 2
    length = len(countour_list)
    for j in range(length):
        start = countour_list[j]
        end = countour_list[(j + 1) % length]
        pt1 = (landmarks[start, 0], landmarks[start, 1])
        pt2 = (landmarks[end, 0], landmarks[end, 1])
        cv2.line(img, pt1, pt2, color=c, thickness=thickness)


def draw_piecewise_linear_list(img, landmarks, countour_list, c, thickness=1):
    if drawthickline is True:
        thickness = 2
    length = len(countour_list)
    for j in range(length - 1):
        start = countour_list[j]
        end = countour_list[(j + 1)]
        pt1 = (landmarks[start, 0], landmarks[start, 1])
        pt2 = (landmarks[end, 0], landmarks[end, 1])
        cv2.line(img, pt1, pt2, color=c, thickness=thickness)


def draw_piecewise_linear_curve(img, landmarks, start_idx, end_idx, c, thickness=1):
    if drawthickline is True:
        thickness = 2
    for j in range(start_idx, end_idx):
        pt1 = (landmarks[j, 0], landmarks[j, 1])
        pt2 = (landmarks[j + 1, 0], landmarks[j + 1, 1])
        cv2.line(img, pt1, pt2, color=c, thickness=thickness)


def draw_landmarks(img, landmarks):
    img = img.copy()
    for i in range(img.shape[0]):
        if landmarks.shape[1] == 159:

            draw_piecewise_linear_curve(img[i], landmarks[i], 0, 16, [1, 0, 0])               # jaw line
            draw_piecewise_linear_curve(img[i], landmarks[i], 55, 69, [0, 1, 0])              # nose line
            draw_piecewise_linear_curve(img[i], landmarks[i], 51, 54, [0, 1, 0])              # nose ridge

            draw_piecewise_linear_curve(img[i], landmarks[i], 151, 154, [0, 0, 1])              # eight line
            draw_piecewise_linear_curve(img[i], landmarks[i], 155, 158, [0, 0, 1])              # eight line

            draw_contour(img[i], landmarks[i], 17, 33, [1, 0, 1])                             # left eyebrow
            draw_contour(img[i], landmarks[i], 34, 50, [1, 0, 1])                             # right eyebrow

            draw_contour(img[i], landmarks[i], 70, 81, [1, 1, 0])                     # left eye
            draw_contour(img[i], landmarks[i], 82, 93, [1, 1, 0])                    # right eye
            draw_contour(img[i], landmarks[i], 94, 117, [1, 0, .5])                  # outer lip

            draw_contour(img[i], landmarks[i], 133, 140, [0, 1, 1])                     # eye ball
            draw_contour(img[i], landmarks[i], 142, 149, [0, 1, 1])

            #draw_contour_list( img[i], landmarks[i], [66, 78, 79, 80, 72, 81, 82, 83], [0.0, 1.0, 1.0] )
            draw_piecewise_linear_list(img[i], landmarks[i], [94, 118, 119, 120, 121, 122, 123, 124, 106], [0.0, 1.0, 1.0])
            draw_piecewise_linear_list(img[i], landmarks[i], [106, 125, 126, 127, 128, 129, 130, 131, 94], [0.0, 1.0, 0.0])

            draw_contour_list(img[i], landmarks[i], [132, 141, 150], [1.0, 1.0, 1.0])

        else:
            draw_piecewise_linear_curve(img[i], landmarks[i], 0, 32, [1, 0, 0])               # jaw line
            draw_piecewise_linear_curve(img[i], landmarks[i], 74, 92, [0, 1, 0])              # nose line
            draw_piecewise_linear_curve(img[i], landmarks[i], 67, 73, [0, 1, 0])              # nose ridge

            draw_piecewise_linear_curve(img[i], landmarks[i], 190, 193, [0, 0, 1])              # eight line
            draw_piecewise_linear_curve(img[i], landmarks[i], 194, 197, [0, 0, 1])              # eight line

            draw_contour(img[i], landmarks[i], 33, 49, [1, 0, 1])                             # left eyebrow
            draw_contour(img[i], landmarks[i], 50, 66, [1, 0, 1])                             # right eyebrow

            draw_contour(img[i], landmarks[i], 93, 104, [1, 1, 0])                     # left eye
            draw_contour(img[i], landmarks[i], 105, 116, [1, 1, 0])                    # right eye
            draw_contour(img[i], landmarks[i], 117, 140, [1, 0, .5])                  # outer lip

            draw_contour(img[i], landmarks[i], 155, 170, [0, 1, 1])                     # eye ball
            draw_contour(img[i], landmarks[i], 172, 187, [0, 1, 1])

            #draw_contour_list( img[i], landmarks[i], [66, 78, 79, 80, 72, 81, 82, 83], [0.0, 1.0, 1.0] )

            draw_piecewise_linear_list(img[i], landmarks[i], [117] + list(range(117, 148)) + [129], [0.0, 1, 1])
            draw_piecewise_linear_list(img[i], landmarks[i], [129] + list(range(148, 155)) + [117], [0.0, 1, 0.0])
    return img


def draw_landmarks_uint8(img, landmarks, draw_point=False, draw_tongue=False, weights_for_landmark_vertices=None):
    img = img.copy()

    if landmarks.dtype == np.float32 or landmarks.dtype == np.float64:
        landmarks = landmarks.astype(np.int32)

    for i in range(img.shape[0]):

        if landmarks.shape[1] == 159:
            draw_piecewise_linear_curve(img[i], landmarks[i], 0, 16, [255, 0, 0])               # jaw line
            draw_piecewise_linear_curve(img[i], landmarks[i], 55, 69, [0, 255, 0])              # nose line
            draw_piecewise_linear_curve(img[i], landmarks[i], 51, 54, [0, 255, 0])              # nose ridge

            draw_piecewise_linear_curve(img[i], landmarks[i], 151, 154, [0, 0, 255])              # eight line
            draw_piecewise_linear_curve(img[i], landmarks[i], 155, 158, [0, 0, 255])              # eight line

            draw_contour(img[i], landmarks[i], 17, 33, [255, 0, 255])                             # left eyebrow
            draw_contour(img[i], landmarks[i], 34, 50, [255, 0, 255])                             # right eyebrow

            draw_contour(img[i], landmarks[i], 70, 81, [255, 255, 0])                     # left eye
            draw_contour(img[i], landmarks[i], 82, 93, [255, 255, 0])                    # right eye
            draw_contour(img[i], landmarks[i], 94, 117, [255, 0, 127])                  # outer lip

            draw_contour(img[i], landmarks[i], 133, 140, [0, 255, 255])                     # eye ball
            draw_contour(img[i], landmarks[i], 142, 149, [0, 255, 255])

            #draw_contour_list( img[i], landmarks[i], [66, 78, 79, 80, 72, 81, 82, 83], [0.0, 1.0, 1.0] )
            draw_piecewise_linear_list(img[i], landmarks[i], [94, 118, 119, 120, 121, 122, 123, 124, 106], [0.0, 255, 255])
            draw_piecewise_linear_list(img[i], landmarks[i], [106, 125, 126, 127, 128, 129, 130, 131, 94], [0.0, 255, 0.0])

        else:
            draw_piecewise_linear_curve(img[i], landmarks[i], 0, 32, [255, 0, 0])               # jaw line
            draw_piecewise_linear_curve(img[i], landmarks[i], 74, 92, [0, 255, 0])              # nose line
            draw_piecewise_linear_curve(img[i], landmarks[i], 67, 73, [0, 255, 0])              # nose ridge

            draw_piecewise_linear_curve(img[i], landmarks[i], 190, 193, [0, 0, 255])              # eight line
            draw_piecewise_linear_curve(img[i], landmarks[i], 194, 197, [0, 0, 255])              # eight line

            draw_contour(img[i], landmarks[i], 33, 49, [255, 0, 255])                             # left eyebrow
            draw_contour(img[i], landmarks[i], 50, 66, [255, 0, 255])                             # right eyebrow

            draw_contour(img[i], landmarks[i], 93, 104, [255, 255, 0])                     # left eye
            draw_contour(img[i], landmarks[i], 105, 116, [255, 255, 0])                    # right eye
            draw_contour(img[i], landmarks[i], 117, 140, [255, 0, 127])                  # outer lip

            draw_contour(img[i], landmarks[i], 155, 170, [0, 255, 255])                     # eye ball
            draw_contour(img[i], landmarks[i], 172, 187, [0, 255, 255])

            #draw_contour_list( img[i], landmarks[i], [66, 78, 79, 80, 72, 81, 82, 83], [0.0, 1.0, 1.0] )

            draw_piecewise_linear_list(img[i], landmarks[i], [117] + list(range(117, 148)) + [129], [0.0, 255, 255])
            draw_piecewise_linear_list(img[i], landmarks[i], [129] + list(range(148, 155)) + [117], [0.0, 255, 0.0])

    return img
