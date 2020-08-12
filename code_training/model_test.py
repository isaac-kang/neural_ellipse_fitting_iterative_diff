import tensorflow as tf
import numpy as np
import cv2

import sys
import os
from time import *
import input_stream
sys.path.append(os.path.join(os.getcwd(), "code_commons"))
from global_constants import *
from auxiliary_ftns import *
import ntpath

save_result_image = True

def run_test(sess, cdnet, FLAGS, mode='folder', frame_size=None, srcname=None,
             wait_time=1, save_video=False, out_video_file_name=None, fps=24.0,
             ckpt_basename=None, evaluation=True):
    if evaluation is True:
        dirpath = os.path.split(srcname)[0]

    if save_video is True:
        assert ckpt_basename is not None

    font = cv2.FONT_HERSHEY_SIMPLEX

    if mode == 'folder':
        assert srcname is not None
        frame = input_stream.Frame('folder', srcname)
        if frame_size is not None:
            frame.set_size(frame_size)
    else:
        raise Exception('Wrong Test Mode')

    if save_video is True:
        if out_video_file_name is None:
            filename = strftime("auto_generated_name_%d%b%Y_%H_%M", gmtime()) + ckpt_basename + ".avi"
        else:
            filename = out_video_file_name
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, fps, (IMAGE_WIDTH, IMAGE_HEIGHT * 3))

    counter = 0

    while True:
        # print( f'image counter = {counter}')
        counter = counter + 1
        tic = time()
        image, filename = frame.get_frame()

        if image is None:
            break

        # Create input
        network_input = cv2.resize(image[:, :, ::-1], (IMAGE_WIDTH, IMAGE_HEIGHT))
        gray = cv2.cvtColor(network_input, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edge_map = cv2.Canny(gray.astype(np.uint8), 20, 100)
        edge_map = np.expand_dims(edge_map, axis=-1)
        edge_map = (edge_map / 255.0 - 0.5) * np.sqrt(2.0)
        src = network_input.astype(np.float32) / 255.0
        network_input = (src.copy() - 0.5) * np.sqrt(2.0)
        mask_dummy = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=np.uint8)
        mask_dummy = cv2.ellipse(mask_dummy, (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2), (IMAGE_WIDTH // 3, IMAGE_HEIGHT // 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
        mask_dummy = mask_dummy.astype(np.float32) / 255.0
        mask_dummy = np.clip(mask_dummy * 1000, 0, 255.0)
        mask_dummy = mask_dummy[:, :, 0].astype(np.float32) / 255.0
        network_input = np.concatenate((network_input, edge_map, mask_dummy), axis=-1)
        network_input = np.array(network_input).reshape([1, IMAGE_HEIGHT, IMAGE_WIDTH, 5])


        # Feedforward
        _output, _mask = sess.run([cdnet.output, cdnet.output_mask], feed_dict={cdnet.input: network_input})
        _output = _output[0]
        _mask = _mask[0]

        # Save image
        _mask = np.expand_dims(_mask, axis=-1)
        _mask = np.tile(_mask, [1, 1, 3])
        overlap = src.copy() * 0.5
        overlap[:, :, 0] = overlap[:, :, 0] + _mask[:, :, 0]
        rst = np.concatenate([src, overlap, _mask], axis=0)
        tmp = rst[:, :, ::-1]
        tmp = np.clip(tmp, 0, 1)
        tmp = (tmp * 255).astype(np.uint8)
        save_img = np.concatenate([src, overlap], axis=1)
        save_img = save_img[:, :, ::-1]
        save_img = np.clip(save_img, 0, 1)
        save_img = (save_img * 255).astype(np.uint8)

        if save_result_image:
            folder_name, image_name = filename.split('/')[-2:]
            if not os.path.exists(os.getcwd() + '/test_results/' + folder_name):
                os.makedirs(os.getcwd() + '/test_results/' + folder_name)
            img_save_dir = os.getcwd() + '/test_results/' + folder_name + '/' + image_name
            print(img_save_dir)
            cv2.imwrite(img_save_dir, save_img)

        if save_video is True:
            out.write(tmp)

    if save_video is True:
        out.release()



if __name__ == "__main__":
    pass