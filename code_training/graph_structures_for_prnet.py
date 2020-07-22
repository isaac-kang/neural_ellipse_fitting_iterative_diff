import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope

from tensorflow.contrib import slim
from tensorflow.contrib.layers import xavier_initializer


def resBlock(x,
             num_outputs, 
             kernel_size = 4, 
             stride=1, 
             activation_fn=tf.nn.relu, 
             normalizer_fn=tcl.batch_norm, 
             scope=None):
    assert num_outputs%2==0 #num_outputs must be divided by channel_factor(2 here)
    with tf.variable_scope(scope, 'resBlock'):
        shortcut = x
        if stride != 1 or x.get_shape()[3] != num_outputs:
            shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride, 
                        activation_fn=None, normalizer_fn=None, scope='shortcut')
        x = tcl.conv2d(x, num_outputs/2, kernel_size=1, stride=1, padding='SAME')
        x = tcl.conv2d(x, num_outputs/2, kernel_size=kernel_size, stride=stride, padding='SAME')
        x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1, activation_fn=None, padding='SAME', normalizer_fn=None)

        x += shortcut       
        x = normalizer_fn(x)
        x = activation_fn(x)

    return x


class resfcn256(object):
    def __init__(self, 
                 FLAGS, 
                 resolution_inp = 256, 
                 resolution_op = 256, 
                 channel = 3, 
                 num_output_channel = 3,
                 scope_name = 'resfcn256'):
        self.scope_name = scope_name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.num_output_channel = num_output_channel

    def build_graph(self, 
                    x, 
                    reuse,
                    is_training = True, final_sigmoid = True):
        with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu, 
                                                                   normalizer_fn=tcl.batch_norm, 
                                                                   biases_initializer=None, 
                                                                   padding='SAME',
                                                                   weights_regularizer=tcl.l2_regularizer(0.0002) ):
                

                    bilinear_interpolation = True 
                    kernel_size = 3

                    size = 16  
                    # x: s x s x 3
                    se = tcl.conv2d(x, num_outputs=size, kernel_size=kernel_size, stride=1) # 256 x 256 x 16
                    se = resBlock(se, num_outputs=size * 2, kernel_size=kernel_size, stride=2) # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 2, kernel_size=kernel_size, stride=1) # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 4, kernel_size=kernel_size, stride=2) # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 4, kernel_size=kernel_size, stride=1) # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 8, kernel_size=kernel_size, stride=2) # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 8, kernel_size=kernel_size, stride=1) # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 16, kernel_size=kernel_size, stride=2) # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 16, kernel_size=kernel_size, stride=1) # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 32, kernel_size=kernel_size, stride=2) # 8 x 8 x 512
                    se = resBlock(se, num_outputs=size * 32, kernel_size=kernel_size, stride=1) # 8 x 8 x 512



                    pd = tcl.conv2d(se, size * 32, kernel_size, stride=1) # 8 x 8 x 512 

                    if bilinear_interpolation is True:
                        pd = tf.image.resize_bilinear(pd, (16,16) )
                        #pd = tf.image.resize_nearest_neighbor(pd, (16,16) )
                        pd = tcl.conv2d(pd, size * 16, kernel_size, stride=1) # 16 x 16 x 256 
                        pd = tcl.conv2d(pd, size * 16, kernel_size, stride=1) # 16 x 16 x 256 
                        pd = tcl.conv2d(pd, size * 16, kernel_size, stride=1) # 16 x 16 x 256                         
                    else:
                        pd = tcl.conv2d_transpose(pd, size * 16, kernel_size, stride=2) # 16 x 16 x 256 
                        pd = tcl.conv2d_transpose(pd, size * 16, kernel_size, stride=1) # 16 x 16 x 256 
                        pd = tcl.conv2d_transpose(pd, size * 16, kernel_size, stride=1) # 16 x 16 x 256 


                    if bilinear_interpolation is True:
                        pd = tf.image.resize_bilinear(pd, (32,32) )
                        #pd = tf.image.resize_nearest_neighbor(pd, (32,32) )
                        pd = tcl.conv2d(pd, size * 8, kernel_size, stride=1) 
                        pd = tcl.conv2d(pd, size * 8, kernel_size, stride=1) # 32 x 32 x 128 
                        pd = tcl.conv2d(pd, size * 8, kernel_size, stride=1) # 32 x 32 x 128 

                    else:
                        pd = tcl.conv2d_transpose(pd, size * 8, kernel_size, stride=2) # 32 x 32 x 128 
                        pd = tcl.conv2d_transpose(pd, size * 8, kernel_size, stride=1) # 32 x 32 x 128 
                        pd = tcl.conv2d_transpose(pd, size * 8, kernel_size, stride=1) # 32 x 32 x 128 


                    if bilinear_interpolation is True:
                        pd = tf.image.resize_bilinear(pd, (64,64) )
                        #pd = tf.image.resize_nearest_neighbor(pd, (64,64) )
                        pd = tcl.conv2d(pd, size * 4, kernel_size, stride=1) 
                        pd = tcl.conv2d(pd, size * 4, kernel_size, stride=1) # 64 x 64 x 64 
                        pd = tcl.conv2d(pd, size * 4, kernel_size, stride=1) # 64 x 64 x 64                         
                    else:
                        pd = tcl.conv2d_transpose(pd, size * 4, kernel_size, stride=2) # 64 x 64 x 64 
                        pd = tcl.conv2d_transpose(pd, size * 4, kernel_size, stride=1) # 64 x 64 x 64 
                        pd = tcl.conv2d_transpose(pd, size * 4, kernel_size, stride=1) # 64 x 64 x 64 
                    

                    if bilinear_interpolation is True:
                        pd = tf.image.resize_bilinear(pd, (128,128) )
                        #pd = tf.image.resize_nearest_neighbor(pd, (128,128) )
                        pd = tcl.conv2d(pd, size * 2, kernel_size, stride=1) 
                        pd = tcl.conv2d(pd, size * 2, kernel_size, stride=1) # 128 x 128 x 32
                    else:
                        pd = tcl.conv2d_transpose(pd, size * 2, kernel_size, stride=2) # 128 x 128 x 32
                        pd = tcl.conv2d_transpose(pd, size * 2, kernel_size, stride=1) # 128 x 128 x 32


                    if bilinear_interpolation is True:
                        pd = tf.image.resize_bilinear(pd, (256,256) )
                        #pd = tf.image.resize_nearest_neighbor(pd, (256,256) )
                        pd = tcl.conv2d(pd, size, kernel_size, stride=1) 
                        pd = tcl.conv2d(pd, size, kernel_size, stride=1) # 256 x 256 x 16
                        pd = tcl.conv2d(pd, self.num_output_channel, kernel_size, stride=1) # 256 x 256 x 3
                        pd = tcl.conv2d(pd, self.num_output_channel, kernel_size, stride=1) # 256 x 256 x 3

                        if final_sigmoid == True:
                            pos = tcl.conv2d(pd, self.num_output_channel, kernel_size, stride=1, activation_fn=tf.nn.sigmoid)                                                    
                        else:
                            pos = tcl.conv2d(pd, self.num_output_channel, kernel_size, stride=1, activation_fn=None) #tf.nn.sigmoid)                        
                        
                    else:
                        pd = tcl.conv2d_transpose(pd, size, kernel_size, stride=2) # 256 x 256 x 16
                        pd = tcl.conv2d_transpose(pd, size, kernel_size, stride=1) # 256 x 256 x 16
                        pd = tcl.conv2d_transpose(pd, self.num_output_channel, kernel_size, stride=1) # 256 x 256 x 3
                        pd = tcl.conv2d_transpose(pd, self.num_output_channel, kernel_size, stride=1) # 256 x 256 x 3
                        pos = tcl.conv2d_transpose(pd, self.num_output_channel, kernel_size, stride=1, activation_fn=tf.nn.sigmoid)


                    return 1.1 *(pos-0.5) + 0.5

