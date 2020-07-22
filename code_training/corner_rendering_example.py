import numpy as np 
import tensorflow as tf 
import cv2

import os, sys 
sys.path.append( os.path.join(os.getcwd(), "code_commons") )
from global_constants import *  
from auxiliary_ftns import *

import dirt_utilities


def orthographic_example():

    pts = []
    pts.append( 0*np.ones( shape=[2], dtype = np.float32 ))
    for i in range(1,MAX_NUM_CORNERS):
        x = 100 * np.cos( 2 * np.pi * i / (MAX_NUM_CORNERS-1) ) + pts[0][0]
        y = 100 * np.sin( 2 * np.pi * i / (MAX_NUM_CORNERS-1) ) + pts[0][1]
        pt = np.array( [x,y], dtype = np.float32 )
        pts.append( pt )

    pts = np.array( pts )
    pts = np.reshape( pts, [1,-1,2])
    vertices_batch = pts
    vertices = tf.constant( vertices_batch, dtype = tf.float32)
    
    #reflectances_batch = np.array( [ [[1,0,0], [0,1,0], [0,0,1], [1,1,0] ], [[1,0,0], [0,1,0], [0,0,1], [1,1,0] ] ] , dtype = np.float32 )
    #reflectances_batch = np.ones_like( vertices_batch )
    triangles = [ [0,1,2], [0,2,3], [0,3,4], [0,4,5], [0,5,6], [0,6,1] ] 
    triangles_np = np.array( triangles, dtype = np.int32 ) 

    print( vertices_batch.shape )


    #reflectances = tf.constant( reflectances_batch, dtype = tf.float32)
    trinagles = tf.constant(triangles, dtype = tf.int32 )

    frame_width = 256
    frame_height = 256
    cx = 128
    cy = 128
    output_dictionary = dirt_utilities.dirt_rendering_orthographic( vertices, trinagles, reflectances = None, frame_width = frame_width, frame_height = frame_height, cx = cx, cy = cy  )

    with tf.Session() as sess:
        _rst0, _rst1 = sess.run( [output_dictionary["rendering_results"], output_dictionary["vertices"] ] )
        _rst0 = _rst0[0]    


        for j in range( triangles_np.shape[0] ):
            u, v, w = triangles_np[j]

            pt1 = ( int(vertices_batch[0,u,0]  + cx) , int(-vertices_batch[0,u,1] + cy) )
            pt2 = ( int(vertices_batch[0,v,0]  + cx) , int(-vertices_batch[0,v,1] + cy) )
            pt3 = ( int(vertices_batch[0,w,0]  + cx) , int(-vertices_batch[0,w,1] + cy) )
            
            cv2.line( _rst0, pt1, pt2, color=(1,0,0), thickness = 3 )
            cv2.line( _rst0, pt2, pt3, color=(1,0,0), thickness = 3 )
            cv2.line( _rst0, pt3, pt1, color=(1,0,0), thickness = 3 )


        cv2.imshow( "rst0", _rst0 )
        cv2.waitKey(0)


if __name__ == "__main__":
    orthographic_example()
    #perspective_example()

    os._exit(0)