import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope


def resBlock_a(x,
             kernel_size=3, 
             num_outputs =[32,32,128], 
             stride=1, 
             activation_fn=tf.nn.relu, 
             normalizer_fn=tcl.batch_norm, 
             scope=None):

    with tf.variable_scope(scope, 'resBlock_a'):

        x = normalizer_fn(x)
        x = activation_fn(x)
        
        shortcut = tcl.conv2d(x, num_outputs[2], kernel_size=1, stride=stride, 
                        activation_fn=None, normalizer_fn=None, scope='shortcut')
            
        x = tcl.conv2d(x, num_outputs[0], kernel_size=1, stride=1, padding='SAME')
        x = tcl.conv2d(x, num_outputs[1], kernel_size=kernel_size, stride=stride, padding='SAME')
        x = tcl.conv2d(x, num_outputs[2], kernel_size=1, stride=1, activation_fn=None, padding='SAME', normalizer_fn=None)
        x += shortcut       
        
    return x


def resBlock_b(x,
            kernel_size=3, 
            num_outputs =[64,64,256], 
            stride=1, 
            activation_fn=tf.nn.relu, 
            normalizer_fn=tcl.batch_norm, 
            last_normalizer_fn=None,
            scope=None):

    with tf.variable_scope(scope, 'resBlock_b'):

        shortcut = x
#        print( x.get_shape()[3], num_outputs[2])
        assert x.get_shape()[3] == num_outputs[2]
        
        x = normalizer_fn(x)
        x = activation_fn(x)
    
        x = tcl.conv2d(x, num_outputs[0], kernel_size=1, stride=1, padding='SAME')
        x = tcl.conv2d(x, num_outputs[1], kernel_size=kernel_size, stride=stride, padding='SAME')
        x = tcl.conv2d(x, num_outputs[2], kernel_size=1, stride=1, activation_fn=None, padding='SAME', normalizer_fn=None)
        x += shortcut      
        
        if last_normalizer_fn is not None:
            x = last_normalizer_fn(x)
        
        return x


def pool(input, name=None, kernel=[1,2,2,1], stride=[1,2,2,1]):
    pool = tf.nn.max_pool(input, kernel, stride, 'SAME', name=name)
#    print(f"{name} pool: {pool.get_shape()}")
    return pool

def hourglass( input, n ):

    pool1 = pool(input)    
    low1 = resBlock_b( pool1 )
    
    up1 = resBlock_b( input )    
    
    # recursive
    if n > 1:
        low2 = hourglass( low1, n - 1 )
    else:
        low2 = resBlock_b( low1 )
        
    low3 = resBlock_b(low2 )

    with tf.variable_scope('resize'):
        up_size = tf.shape(up1)[1:3]
        up2 = tf.image.resize_bilinear(low3, up_size)
    
    with tf.variable_scope('sum'):
        up = up1 + up2
        
    return up


def build_graph( input, is_training, num_stacks, num_recursion_each_hg, num_heatmap_channels ):
    heatmaps = []
    with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
        with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu, 
                                                        normalizer_fn=tcl.batch_norm, 
                                                        biases_initializer=None, 
                                                                padding='SAME'):
            with tf.variable_scope('head') as scope:    
                x = tcl.conv2d(input, num_outputs=32, kernel_size=7, stride=2, \
                            activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None)
                x = resBlock_a( x )
                x = pool(x)
                x = resBlock_b( x, num_outputs =[32,32,128])
                x = resBlock_a( x, num_outputs =[64,64,256])

            inter = x
        
            for i in range(num_stacks):
                with tf.variable_scope('stack_%s'%i):
                    with tf.variable_scope('hourglass'):
                        hg = hourglass( inter, num_recursion_each_hg ) 

                    y = resBlock_b(hg, last_normalizer_fn=tcl.batch_norm )
                    y = tcl.conv2d(y, num_outputs=256, kernel_size=1, activation_fn=tf.nn.relu, padding='SAME',\
                                normalizer_fn=tcl.batch_norm)

                    heatmap = tcl.conv2d(y, num_outputs=num_heatmap_channels, kernel_size=1, activation_fn=None, padding='SAME', \
                                    normalizer_fn=None)
                    heatmaps += [heatmap]

                    # Residual link across hourglasses
                    if i < num_stacks-1:
                        with tf.variable_scope('interlink%s'%(i)):
                            inter = inter \
                                + tcl.conv2d(y, num_outputs=256, kernel_size=1, activation_fn=None,\
                                            padding='SAME', normalizer_fn=None) \
                                    + tcl.conv2d(heatmap, num_outputs=256, kernel_size=1, activation_fn=None,\
                                                padding='SAME', normalizer_fn=None) 
    return heatmaps 