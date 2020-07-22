import numpy as np 
import tensorflow as tf 
import cv2
from tensorflow.python.framework import ops


import dirt_utilities


def scale_compensation( parameter_vectors_tf):
    bias = [ 0, 0, 0, 0.5, 0.5 ]
    wght = [ 600, 600, 4, 100, 100 ]
    offset = [ 300, 300, 0, 0, 0 ]



    wght = np.array( wght, dtype = np.float32 )
    wght = tf.constant( wght, dtype = tf.float32 )

    bias = np.array( bias, dtype = np.float32 )
    bias = tf.constant( bias, dtype = tf.float32 )        

    offset = np.array( offset, dtype = np.float32 )
    offset = tf.constant( offset, dtype = tf.float32 )   

    parameter_vectors_tf = ( tf.math.sigmoid( parameter_vectors_tf ) -0.5 )    

    scaled_parameter_vectors_tf = (parameter_vectors_tf + bias ) * wght + offset 
    return scaled_parameter_vectors_tf


def transformation( cx, cy, theta, lambda1, lambda2, name=None):

    # x is indexed by *, x/y/z
    with ops.name_scope(name, 'transformation', []) as scope:
        #x = tf.convert_to_tensor(cx, name='x')
        zeros = tf.zeros_like(cx)  # indexed by *
        ones = tf.ones_like(zeros)

        elements = [
            [ lambda1 * tf.cos(theta), lambda2 * tf.sin(theta), cx ],
            [  -lambda1 * tf.sin(theta), lambda2 * tf.cos(theta), cy ],
            [zeros, zeros, ones ] ]

        return tf.squeeze( tf.transpose( tf.convert_to_tensor(elements, dtype=tf.float32)  ) )
        
        """
        return tf.stack([
            tf.stack([ lambda1 * tf.cos(theta), - lambda2 * tf.sin(theta), cx ], axis=-1),  # indexed by *, x/y/z (out)
            tf.stack([ lambda1 * tf.sin(theta), lambda2 * tf.cos(theta), cy ], axis=-1),
            tf.stack([zeros, zeros, ones ], axis=-1)
        ], axis=-2)  # indexed by *, x/y/z/w (in), x/y/z/w (out)
        """


def point_plot( img, pts ):
    pts = pts.copy()
    pts = pts.astype ( np.int32)
    num_batch = img.shape[0]
    num_pts = pts.shape[1]

    for i in range( num_batch ):
        for j in range( num_pts ):
            x1,y1 = pts[i,j,0], pts[i,j,1]
            x2,y2 = pts[i,(j+1)%num_pts,0], pts[i,(j+1)%num_pts,1]
            cv2.line( img[i], (x1,y1), (x2,y2), (255,0,0), 3 )

if __name__ == "__main__":
    img = cv2.imread( './data/s00000.png' )
    mask = cv2.imread( './data/e00000.png')

    cv2.imshow( "wnd", img )
    cv2.imshow( "mask", mask )

    cv2.waitKey(1)

    IMAGE_WIDTH = img.shape[1]
    IMAGE_HEIGHT = img.shape[0]


    NUM_BDRY_POINTS = 20
    triangles = []
    for i in range(NUM_BDRY_POINTS):
        triangles.append( [i,(i+1)%NUM_BDRY_POINTS,NUM_BDRY_POINTS] )

    trinagles = tf.constant(triangles, dtype = tf.int32 )


    # x, y, theta, lambda_1, lambda_2 
    batch_slice = 7
    initial_value = np.random.normal( size=(batch_slice,5) )    


    img = np.zeros( shape = (batch_slice,IMAGE_HEIGHT,IMAGE_WIDTH,3), dtype = np.uint8 )


    parameter_vectors_stf = tf.Variable( initial_value, dtype = tf.float32 )
    parameter_vectors_tf = scale_compensation(parameter_vectors_stf)    

    mask = mask[:,:,0]
    mask = mask.astype( np.float32 ) / 255.0

    centerslice = np.mean( np.array( np.where( mask != 0 ) ), axis=1 ).astype( np.int32 )
    centerslice = centerslice[::-1]
    centerslice = np.tile( centerslice, [batch_slice, 1])
    centerslice = tf.constant( centerslice, dtype = tf.float32 )


    mask = np.expand_dims( mask, axis = 0)
    mask_slice = np.tile( mask, [batch_slice, 1, 1])
    mask_slice = tf.constant( mask_slice, dtype = tf.float32 )


    centerx = parameter_vectors_tf[:,0:1]
    centery = parameter_vectors_tf[:,1:2]
    angle  = parameter_vectors_tf[:,2:3] 
    radius1 = parameter_vectors_tf[:,3:4] 
    radius2 = parameter_vectors_tf[:,4:5] 

    M = np.zeros( shape = [1, NUM_BDRY_POINTS+1,3], dtype = np.float32 )
    for i in range( NUM_BDRY_POINTS ):
        M[0,i,0] = np.cos( 2*np.pi*i/NUM_BDRY_POINTS )
        M[0,i,1] = np.sin( 2*np.pi*i/NUM_BDRY_POINTS )
        M[0,i,2] = 1.0 
    M[0,NUM_BDRY_POINTS,0] = M[0,NUM_BDRY_POINTS,1] = 0.0 
    M[0,NUM_BDRY_POINTS,2] = 1      # center 

    M = tf.constant( M, tf.float32 )
    M = tf.tile( M, [batch_slice, 1, 1])

    T = transformation( centerx, centery, angle, radius1, radius2 )
    M = tf.matmul( M, T )

    frame_width = IMAGE_WIDTH
    frame_height = -IMAGE_HEIGHT
    cx = 0
    cy = IMAGE_HEIGHT
    output_dictionary = dirt_utilities.dirt_rendering_orthographic( M, trinagles, reflectances = None, frame_width = frame_width, frame_height = frame_height, cx = cx, cy = cy  )
    rendering_result =  output_dictionary["rendering_results"][...,0]

    error_map = tf.abs( rendering_result  - mask_slice ) 
    region_loss = 100.0 * tf.reduce_mean( error_map  )

    center_points = M[:,-1,:2]
    loss = 1 * tf.reduce_mean( tf.abs(center_points - centerslice) ) + region_loss 

    learning_rate = .01
    global_step = tf.Variable( 0, trainable=False) 
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

    update_var_list = tf.get_collection( key = tf.GraphKeys.TRAINABLE_VARIABLES )
    #print_var_status( update_var_list )

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss,global_step=global_step )

    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer())

        for i in range(4000):
            sess.run( train_op )

            if i%100 == 0:
                _loss, _error_map, _center_points, _centerslice = sess.run( [loss, error_map, center_points, centerslice ])
                print(_loss)
                print( _center_points[0], _centerslice[0] )

                cv2.imshow( "_error_map", _error_map[0] )                
                cv2.waitKey(1)

        _M = sess.run( M )
        _rendering_result = sess.run( rendering_result )

        point_plot( img, _M )

        cv2.imshow( "wnd", img[0] )
        cv2.imshow( "rendering", _rendering_result[0] )        
        cv2.waitKey(0)
        print( _M )

    """
    positions = shape_vector * M 

    vertices_batch = tf.reshape( positions, [-1,NUM_BDRY_POINTS+1,3])
    reflectances_batch = tf.ones_like( vertices_batch )
    """


