import os
import json
import sys
import numpy as np
import numpy.random as random
import cv2
import tqdm
import matplotlib.pyplot as plt
import math
import time

sys.path.append(os.path.join(os.getcwd(), "code_commons"))
from global_constants import *
from auxiliary_ftns import *


def iterateTrainData(img_directories):
    import pickle
    dataset = []
    # List up all files and metadata
    for img_dir in img_directories:
        # Is there 'filelist' pickle?
        sample_set_filename = os.path.join(img_dir, "ellipse_fitting_file_information")
        if os.path.isfile(sample_set_filename) is True:
            with open(sample_set_filename, 'rb') as fp:
                cur_dataset = pickle.load(fp)
            print(f"fileinformation (having {len(cur_dataset)} images) of '{img_directories}'' exists in '{sample_set_filename}', using this file without searches")
        else:
            cur_dataset = []
            for dirpath, _, filenames in os.walk(img_dir):
                # image
                for filename in tqdm.tqdm(filenames):
                    if filename.endswith('.png') or filename.endswith('.jpg'):
                        # ext
                        ext = '.png' if filename.endswith('.png') else '.jpg'
                        # image
                        imgfilename = os.path.join(dirpath, os.path.splitext(filename)[0] + ext)
                        # filename should be number
                        basename = os.path.splitext(filename)[0]
                        idex = int(basename)
                        if os.path.isfile(imgfilename) is True:
                            # img
                            cur_dataset.append((
                                imgfilename
                            ))
            with open(sample_set_filename, 'wb') as fp:
                # save pickle of filelist
                pickle.dump(cur_dataset, fp)
        dataset += cur_dataset
    # list of img in dir
    return dataset


class TrainDataGenerator:

    def __init__(self, img_directory, GEOMETRIC_DISTORTION=True, PHOTOMETRIC_DISTORTION=True):

        # list of img
        self.dataset = iterateTrainData(img_directory)

        # distortion settings
        self.PHOTOMETRIC_DISTORTION = PHOTOMETRIC_DISTORTION
        self.GEOMETRIC_DISTORTION = GEOMETRIC_DISTORTION
        self.noise = True
        self.RING_DISTORTION = True

    def geometric_distortion(self, image_width, image_height):
        # rotation
        rotation = np.random.normal() * np.pi / 12.0    # 15 degrees

        # translation
        translation = np.random.normal(size=[2]) * 10.0

        # scale
        scale = np.exp(np.random.normal() / 10.0)
        scalex = scale * (1 + 0.1 * np.random.uniform(-0.5, 0.8))
        scaley = scale

        if self.GEOMETRIC_DISTORTION is False:
            rotation = 0
            translation = np.zeros_like(translation)
            scalex = scaley = scale = 1

        #  perturbate the image
        T = \
            np.array([
                [1, 0, image_width * 0.5],
                [0, 1, image_height * 0.5],
                [0, 0, 1]], dtype='float32') @ \
            np.array([
                [1, 0, translation[0]],
                [0, 1, translation[1]],
                [0, 0, 1]], dtype='float32') @ \
            np.array([
                [scalex, 0, 0],
                [0, scaley, 0],
                [0, 0, 1]], dtype='float32') @ \
            np.array([
                [np.cos(rotation), np.sin(rotation), 0],
                [-np.sin(rotation), np.cos(rotation), 0],
                [0, 0, 1]], dtype='float32') @ \
            np.array([
                [1, 0, -image_width * 0.5],
                [0, 1, -image_height * 0.5],
                [0, 0, 1]], dtype='float32')

        return T

    def generate_data(self, imgfilename1, imgfilename2):
        # img size
        w = image_width = IMAGE_WIDTH
        h = image_height = IMAGE_HEIGHT
        # img
        try:
            img1 = cv2.resize(cv2.imread(imgfilename1), (h, w), cv2.INTER_LINEAR)
        except:
            print('img1', imgfilename1)
        try:
            img2 = cv2.resize(cv2.imread(imgfilename2), (h, w), cv2.INTER_LINEAR)
        except:
            print('img2', imgfilename2)
        # mask
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask_dummy = np.zeros((h, w, 3), dtype=np.uint8)
        mask_color = (255, 255, 255)
        if self.RING_DISTORTION:
            mask1 = np.zeros((h, w, 3), dtype=np.uint8)
            mask_color1 = (255, 255, 255)
            color_image = np.zeros((h, w, 3), dtype=np.uint8)
            color_image[:, :, 0] = np.random.randint(0, 256)
            color_image[:, :, 1] = np.random.randint(0, 256)
            color_image[:, :, 2] = np.random.randint(0, 256)
        

        # ELLIPSE GENERATION
        ellipse_area = 0
        # ellipse area threshold
        ellipse_area_min = h * w / 5
        ellipse_area_max = h * w * 4 / 5
        while (ellipse_area < ellipse_area_min or ellipse_area > ellipse_area_max):
            # center point uniform around square
            center_x = random.randint(1, w - 2)
            center_y = random.randint(1, h - 2)
            # ellipse not cropped by border
            axis_x_limit = min(abs(center_x - 0), abs(center_x - (w - 1)))
            axis_y_limit = min(abs(center_y - 0), abs(center_y - (h - 1)))
            axis_ratio_limit = 5
            r_limit = min(axis_x_limit, axis_y_limit)
            # set r1 so ellipse is not cropped by border
            axis_x = random.randint(0, r_limit)
            axis_x = 1 if axis_x == 0 else axis_x
            # r2 smaller than r1
            # r2 bigger than r1/5
            axis_y = random.randint(axis_x // axis_ratio_limit, axis_x)
            axis_y = 1 if axis_y == 0 else axis_y
            ellipse_area = np.pi * axis_x * axis_y
        # random theta
        theta = random.randint(0, 360)
        angle_start = 0
        angle_end = 360
        # ellipse
        mask = cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), theta, angle_start, angle_end, mask_color, cv2.FILLED)
        mask_dummy = cv2.ellipse(mask_dummy, (w // 2, h // 2), (w // 3, h // 4), 0, angle_start, angle_end, mask_color, cv2.FILLED)
        mask = mask.astype(np.float32) / 255.0
        mask_dummy = mask_dummy.astype(np.float32) / 255.0
        if self.RING_DISTORTION:
            ring_thickness = random.randint(1, axis_y // 2)
            mask1 = cv2.ellipse(mask1, (center_x, center_y), (axis_x - ring_thickness, axis_y - ring_thickness), 
                theta, angle_start, angle_end, mask_color1, cv2.FILLED)
            mask1 = mask1.astype(np.float32) / 255.0
        # mask center
        mask_center = np.array([center_x, center_y]).astype(np.float32)

        # combine images
        if self.RING_DISTORTION == True and np.random.randint(10) > 5:
            img = np.where(mask > 0, color_image, img2)
            img = np.where(mask1 > 0, img1, img)
        else:
            img = np.where(mask > 0, img2, img1)

        # photometric_distortion
        r = g = b = 0
        if self.PHOTOMETRIC_DISTORTION is True:
            r = np.random.randint(0, 256)
            g = np.random.randint(0, 256)
            b = np.random.randint(0, 256)

        s = np.sqrt(np.float(image_width * image_height) / img.shape[0] / img.shape[1])
        B = np.array([[s, 0, image_width / 2 - s * img.shape[1] // 2], [0, s, image_height / 2 - s * img.shape[0] // 2], [0, 0, 1]], dtype=np.float32)

        # geometric_distortion
        # T = G @ B
        T = self.geometric_distortion(image_width, image_height) @ B

        # apply geometric_distortion
        canvas = cv2.warpAffine(img, T[:2], (image_width, image_height), flags=cv2.INTER_CUBIC, borderValue=(r, g, b))

        # apply geometric_distortion to center
        mask_center = np.array([mask_center[0], mask_center[1], 1.0], dtype=np.float32)
        mask_center_orig = mask_center
        mask_center = (T[:2] @ mask_center).T

        # apply geometric_distortion to axis_x_points
        mask_axis_x_pts_start = np.array([mask_center_orig[0] + axis_x * np.cos(np.deg2rad(theta)),
                                          mask_center_orig[1] + axis_x * np.sin(np.deg2rad(theta)), 1.0], dtype=np.float32)
        mask_axis_x_pts_end = np.array([mask_center_orig[0] - axis_x * np.cos(np.deg2rad(theta)),
                                        mask_center_orig[1] - axis_x * np.sin(np.deg2rad(theta)), 1.0], dtype=np.float32)
        mask_axis_x_pts_start = (T[:2] @ mask_axis_x_pts_start).T
        mask_axis_x_pts_end = (T[:2] @ mask_axis_x_pts_end).T
        mask_axis_x_pts = np.concatenate((mask_axis_x_pts_start, mask_axis_x_pts_end))

        # apply geometric_distortion to axis_y_points
        mask_axis_y_pts_start = np.array([mask_center_orig[0] + axis_y * np.cos(np.deg2rad(theta + 90)),
                                          mask_center_orig[1] + axis_y * np.sin(np.deg2rad(theta + 90)), 1.0], dtype=np.float32)
        mask_axis_y_pts_end = np.array([mask_center_orig[0] - axis_y * np.cos(np.deg2rad(theta + 90)),
                                        mask_center_orig[1] - axis_y * np.sin(np.deg2rad(theta + 90)), 1.0], dtype=np.float32)
        mask_axis_y_pts_start = (T[:2] @ mask_axis_y_pts_start).T
        mask_axis_y_pts_end = (T[:2] @ mask_axis_y_pts_end).T
        mask_axis_y_pts = np.concatenate((mask_axis_y_pts_start, mask_axis_y_pts_end))

        # apply geometric_distortion to annoatation
        # annotated_center = np.array( [annotated_center[0], annotated_center[1], 1.0], dtype = np.float32 )
        # annotated_center = ( T[:2] @ annotated_center).T

        # apply geometric_distortion to mask
        mask = cv2.warpAffine(mask, T[:2], (image_width, image_height), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))

        # flip vertically
        if np.random.randint(2) == 0:
            canvas = canvas[:, ::-1, :]
            mask = mask[:, ::-1, :]
            mask_center[0] = image_width - mask_center[0]
            mask_axis_x_pts[0] = image_width - mask_axis_x_pts[0]
            mask_axis_x_pts[2] = image_width - mask_axis_x_pts[2]
            mask_axis_y_pts[0] = image_width - mask_axis_y_pts[0]
            mask_axis_y_pts[2] = image_width - mask_axis_y_pts[2]

        # flip horizontally
        if np.random.randint(2) == 0:
            canvas = canvas[::-1, :, :]
            mask = mask[::-1, :, :]
            mask_center[1] = image_height - mask_center[1]
            mask_axis_x_pts[1] = image_height - mask_axis_x_pts[1]
            mask_axis_x_pts[3] = image_height - mask_axis_x_pts[3]
            mask_axis_y_pts[1] = image_height - mask_axis_y_pts[1]
            mask_axis_y_pts[3] = image_height - mask_axis_y_pts[3]

        # mask is 0 or 1
        mask = np.clip(mask * 1000, 0, 255.0)
        mask_dummy = np.clip(mask_dummy * 1000, 0, 255.0)
        mask = mask[:, :, 0].astype(np.float32) / 255.0
        mask_dummy = mask_dummy[:, :, 0].astype(np.float32) / 255.0
        
        if self.noise:
            if np.random.randint(3) == 0:
                #   Blur
                blur_radius = int((np.random.normal() * 1.5) ** 2 + 0.5) * 2 + 1
                if blur_radius > 0:
                    canvas = cv2.GaussianBlur(canvas, (blur_radius, blur_radius), 0)

            
            if np.random.randint(3) == 0:
                #   Darken
                canvas = (canvas.astype('float32') * np.random.uniform(0.5, 1)).astype('uint8')


            
            if np.random.randint(3) == 0:
                #   Blocker
                # not being used right now
                margin = 0.1 * IMAGE_WIDTH 
                c_x = np.random.uniform(margin, IMAGE_WIDTH - margin)
                c_y = np.random.uniform(margin, IMAGE_HEIGHT - margin)

                v_x = np.random.normal() * IMAGE_WIDTH / 6.0
                v_y = np.random.normal() * IMAGE_HEIGHT / 6.0
                pts = [
                            (c_x + v_x, c_y + v_y),
                            (c_x - v_y, c_y + v_x),
                            (c_x - v_x, c_y - v_y),
                            (c_x + v_y, c_y - v_x)
                        ]
                pts = np.array([(int(p[0] + 0.5), int(p[1] + 0.5)) for p in pts])
                colors = [
                                (238, 245, 255),
                                (205, 235, 255),
                                (181, 228, 255),
                                (179, 222, 245),
                                (140, 180, 210),
                                (63, 133, 205),
                                (45, 82, 160),
                                (19, 69, 139),
                                (0, 0, 0)
                            ]
                color = colors[np.random.randint(len(colors))]
                cv2.fillConvexPoly(canvas.astype('float32'), pts, color)

        canvas = canvas.astype('float32')
        # # add strength noise
        canvas = canvas * np.random.uniform( 0.5,1.5 )
        # # add uniform noise
        canvas = canvas + np.random.uniform( -50,50 )
        # add gaussian noise
        if np.random.randint(3) == 0:
            noise = np.random.normal(size=[IMAGE_HEIGHT, IMAGE_WIDTH, 3]) * 0.15
            canvas += noise
        image_show = canvas
        # cv2.imwrite(os.getcwd() + '/what/image' + str(np.random.randint(10)) + '.png', image_show)
        # edge map
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edge_map = cv2.Canny(gray.astype(np.uint8), 20, 100)
        edge_map = np.expand_dims(edge_map, axis=-1)
        # cv2.imwrite(os.getcwd() + '/what/edged' + str(np.random.randint(10)) + '.png', edge_map)
        edge_map = (edge_map / 255.0 - 0.5) * np.sqrt(2.0)
        canvas = (canvas / 255.0 - 0.5) * np.sqrt(2.0)
        canvas = canvas[:, :, ::-1]
        # concat with edge map
        canvas = np.concatenate((canvas, edge_map), axis=-1)
        dict = {}
        dict["image"] = canvas # (224, 224, 4)
        dict["mask"] = mask
        dict["mask_dummy"] = mask_dummy
        dict["mask_center"] = mask_center
        dict["mask_axis_x_pts"] = mask_axis_x_pts
        dict["mask_axis_y_pts"] = mask_axis_y_pts
        return dict, True

    def __call__(self):
        dataset = self.dataset

        # random seed
        s = np.uint32(os.getpid() * (np.uint64(time.time()) % 1000))
        np.random.seed(s)
        print('seed:', s)

        try:
            while True:
                # read random file from data dir
                idx = np.random.randint(len(dataset))
                idx1 = idx
                while idx1 == idx:
                    idx1 = np.random.randint(len(dataset))
                imgfilename = dataset[idx]
                imgfilename1 = dataset[idx1]
                train_dict, valid_data = self.generate_data(imgfilename, imgfilename1)
                if valid_data is False:
                    continue
                yield train_dict
        except EOFError:
            return


if __name__ == "__main__":
    pass
