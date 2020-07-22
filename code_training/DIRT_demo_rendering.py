import numpy as np 
import tensorflow as tf 
import pickle
import cv2
from tensorflow.python.framework import ops

import os, sys 
sys.path.append( os.path.join(os.getcwd(), "code_commons") )
from global_constants import *  
from auxiliary_ftns import *

import imageio
import dirt_utilities

uv_parsing_filename = './data/mesh_vertices_landmarks.pickle'
UV_PLANE_WIDTH = UV_PLANE_HEIGHT = 256
NUM_VERTICES = 2687  


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def remove_vertices( triangle_meshes, vertices ):
   new_meshes = []
   triangle_meshes = list( triangle_meshes )
   for element in triangle_meshes:
      if element[0]  in vertices or element[1]  in vertices or element[2] in vertices:
         pass
      else:
         new_meshes.append( element )
   return np.array( new_meshes )


def load_obj( inputfilename):

    vertices = []
    faces = []
    reflectances = []

    with open(inputfilename, 'rb') as fid:
        while True:
            line = fid.readline()
            if not line:
                break

            if line[0] == ord('v'):
                r = g = b = 1
                ############################### 'vertex' ############################### 
                oneline =  line[1:].split()
                if len(oneline) == 6:
                    x, y, z, r, g, b = [float(x) for x in oneline] # read first line
                else:
                    x, y, z = [float(x) for x in oneline] # read first line
                vertices.append( [x,y,z] )
                reflectances.append( [b,g,r])


            elif line[0] == ord('f'):
                ############################### 'face' ############################### 
                a, b, c = [int(x) for x in line[1:].split()] # read first line
                a = a - 1
                b = b - 1
                c = c - 1

                faces.append( [a,b,c] )
            else:
                assert False


    vertices = np.array( vertices ).reshape( [-1,3 ])
    reflectances = np.array( reflectances ).reshape( [-1,3 ])
    faces = np.array( faces ).reshape( [-1,3 ])
    faces = remove_vertices( faces, list(range(2330,2600)) )     # remove eye regions 
    return vertices, reflectances, faces 


def print_ranges( matrix ):
    a, b = matrix.shape
    for i in range(b):
        print( i, ": ", np.min( matrix[:,i]), np.max( matrix[:,i]))


def perspective_example():
    BATCH_SIZE = 3
    FOCAL_LENGTH = 350
    IMAGE_WIDTH = IMAGE_HEIGHT = 512
    CX = IMAGE_WIDTH // 2
    CY = IMAGE_HEIGHT // 2

    movieimages1 = []


    with tf.Session() as sess:
        for i in range(0,60):
            input_vector_ = np.array( [30, 150.0, 0.0, 0.0, 0.000000001, 0,0, -200.0 ]).reshape( [1, -1] )
            intrinsic_vector_ = np.array( [FOCAL_LENGTH, FOCAL_LENGTH, CX, CY])
            input_batch_vector_ = np.tile( input_vector_, [BATCH_SIZE,1])
            intrinsic_vector_ = np.tile( intrinsic_vector_, [BATCH_SIZE,1])

            # X
            input_batch_vector_[0,2] = np.pi/30 * i
            input_batch_vector_[1,3] = np.pi/30 * i
            input_batch_vector_[2,4] = np.pi/30 * i

            # X
            input_batch_vector_[0,5] = 30
            input_batch_vector_[1,5] = 0
            input_batch_vector_[2,5] = 0

            # Y
            input_batch_vector_[0,6] = -30
            input_batch_vector_[1,6] = 30
            input_batch_vector_[2,6] = 30

            # Z
            input_batch_vector_[0,7] = -200
            input_batch_vector_[1,7] = -200
            input_batch_vector_[2,7] = -200
            """
            """

            input_batch_vector = tf.constant(input_batch_vector_ ,tf.float32 )
            intrinsic_vector = tf.constant( intrinsic_vector_, tf.float32 )

            shape_vector = tf.concat( [input_batch_vector[:,0:1], input_batch_vector[:,0:1], input_batch_vector[:,1:2]], axis = 1 )
            shape_vector = tf.expand_dims( shape_vector, axis = 1)
            rotation_params = input_batch_vector[:,2:5]
            trans_params = input_batch_vector[:,5:8]

            POINTS_FOR_DEBUGGING = 4
            M = np.ones( shape = [1, NUM_BDRY_POINTS+1+POINTS_FOR_DEBUGGING,3], dtype = np.float32 )
            for i in range( NUM_BDRY_POINTS ):
                M[0,i,0] = np.cos( 2*np.pi*i/NUM_BDRY_POINTS )
                M[0,i,1] = np.sin( 2*np.pi*i/NUM_BDRY_POINTS )
            M[0,NUM_BDRY_POINTS,:] = 0


            M[0,NUM_BDRY_POINTS+1,0] = 0
            M[0,NUM_BDRY_POINTS+1,1] = 0
            M[0,NUM_BDRY_POINTS+1,2] = 0

            M[0,NUM_BDRY_POINTS+2,0] = 1000
            M[0,NUM_BDRY_POINTS+2,1] = 0
            M[0,NUM_BDRY_POINTS+2,2] = 0
            
            M[0,NUM_BDRY_POINTS+3,0] = 0
            M[0,NUM_BDRY_POINTS+3,1] = 1000
            M[0,NUM_BDRY_POINTS+3,2] = 0

            M[0,NUM_BDRY_POINTS+4,0] = 0
            M[0,NUM_BDRY_POINTS+4,1] = 0
            M[0,NUM_BDRY_POINTS+4,2] = 1000



            M = tf.constant( M, tf.float32 )
            M = tf.tile( M, [BATCH_SIZE, 1, 1])
            positions = shape_vector * M 

            vertices_batch = tf.reshape( positions, [-1,NUM_BDRY_POINTS+POINTS_FOR_DEBUGGING+1,3])

            triangles = []
            for i in range(NUM_BDRY_POINTS):
                triangles.append( [(i+1)%NUM_BDRY_POINTS,i,NUM_BDRY_POINTS] )

            #triangles.append([ NUM_BDRY_POINTS+1,NUM_BDRY_POINTS+2,NUM_BDRY_POINTS+3] )
            #triangles.append([ NUM_BDRY_POINTS+2,NUM_BDRY_POINTS+1,NUM_BDRY_POINTS+3] )
            #triangles.append([ NUM_BDRY_POINTS+4,NUM_BDRY_POINTS+5,NUM_BDRY_POINTS+6] )
            #triangles.append([ NUM_BDRY_POINTS+5,NUM_BDRY_POINTS+4,NUM_BDRY_POINTS+6] )
            #triangles.append([ NUM_BDRY_POINTS+7,NUM_BDRY_POINTS+8,NUM_BDRY_POINTS+9] )
            #triangles.append([ NUM_BDRY_POINTS+8,NUM_BDRY_POINTS+7,NUM_BDRY_POINTS+9] )
            
            triangles_np = np.array( triangles, dtype = np.int32 ) 
            trinagles = tf.constant(triangles, dtype = tf.int32 )

            reflectances_batch = tf.ones_like( vertices_batch )


            vertices = []
            for i in range(NUM_BDRY_POINTS+POINTS_FOR_DEBUGGING+1):
                vertices.append( [i] )

            vertices_indices = tf.constant( vertices, dtype = tf.int32 )


            params_dictionary= {}
            params_dictionary["spherical_harmonics_parameters"] = tf.ones( [BATCH_SIZE,27], dtype=tf.float32)
            params_dictionary["rotation_parameters"] = rotation_params    
            params_dictionary["translation_parameters"] = trans_params
            params_dictionary["convention"] = "option1"
            params_dictionary["camera_rotation_parameters"] = 0 * rotation_params

            params_dictionary["focal_length"] = tf.ones( [BATCH_SIZE], dtype=tf.float32) * NORMALIZED_FOCAL_LENGTH
            params_dictionary["principal_point_cx"] = tf.ones( [BATCH_SIZE], dtype=tf.float32) * IMAGE_WIDTH // 2
            params_dictionary["principal_point_cy"] = tf.ones( [BATCH_SIZE], dtype=tf.float32) * IMAGE_HEIGHT // 2

            output_dictionary = dirt_utilities.rendering( params_dictionary, vertices_batch, reflectances_batch, trinagles, vertices_indices, frame_width = IMAGE_WIDTH, frame_height = IMAGE_HEIGHT )

            """
            positions = tf.concat([positions,tf.ones_like(positions[..., -1:])], axis=-1)    
            rotation_matrix = rodrigues( rotation_params )
            trans_matrix = translation( trans_params )
            K_matrix = intrinsic( intrinsic_vector[:,0], intrinsic_vector[:,1], intrinsic_vector[:,2], intrinsic_vector[:,3])
            projection_matrix = tf.matmul( tf.matmul( trans_matrix, rotation_matrix ), K_matrix )
            projected_points = tf.matmul( positions, projection_matrix )
            """


            _vertices_batch = sess.run( vertices_batch ) 
            _landmarks = sess.run( output_dictionary["landmark_points"] )
            print( _landmarks )        
            
            _rst0, _rst1, _rst2, _rst3, _landmarks  = sess.run( [output_dictionary["full_model_pixels"],   output_dictionary["geometry_model_pixels"], \
                                output_dictionary["depth_maps"], output_dictionary["reflectance_model_pixels"], output_dictionary["landmark_points"]] )

            _rst0 = _rst0[:,:,:,:]
            _rst1 = _rst1[:,:,:,:]
            _rst2 = _rst2[:,:,:]    
            #_rst2 = depth_normalize( 200-_rst2, 0 ,200 )
            _rst2 = (1000-_rst2) /  1000.
            _rst2 = np.expand_dims( _rst2, axis = -1 )
            _rst2 = np.tile( _rst2, [1,1,1,3] )


            color_table = [ (0,0,1), (0,1,0), (1,0,0)]
            for i in range(BATCH_SIZE):
                img = _rst0[i].copy()                

                for j in range(NUM_BDRY_POINTS+1):
                    x0 = int( _landmarks[i,j,0] )
                    y0 = int( _landmarks[i,j,1] )
                    cv2.circle( img, (x0,y0), 1, color=(0,1,1), thickness = -1 )

                for j in range(3):
                    x0 = int( _landmarks[i,NUM_BDRY_POINTS+1,0] )
                    y0 = int( _landmarks[i,NUM_BDRY_POINTS+1,1] )

                    x1 = int( _landmarks[i,NUM_BDRY_POINTS+2+j,0] )
                    y1 = int( _landmarks[i,NUM_BDRY_POINTS+2+j,1] )                    

                    cv2.line( img, (x0,y0), (x1,y1), color_table[j]  )

                _rst0[i] = img

            ny = 1
            nx = BATCH_SIZE
            height = width = IMAGE_WIDTH
            channels = 3
            _rst0 = _rst0.reshape(ny,nx,height,width,channels).transpose(0,2,1,3,4).reshape(height*ny,width*nx,channels)                
            _rst1 = _rst1.reshape(ny,nx,height,width,channels).transpose(0,2,1,3,4).reshape(height*ny,width*nx,channels)
            _rst2 = _rst2.reshape(ny,nx,height,width,channels).transpose(0,2,1,3,4).reshape(height*ny,width*nx,channels)    
            _rst3 = _rst3.reshape(ny,nx,height,width,channels).transpose(0,2,1,3,4).reshape(height*ny,width*nx,channels)
            #_rst2 = np.expand_dims( _rst2, axis = -1 )
            #_rst2 = np.tile( _rst2, [1,1,3])


            rst =  np.concatenate( [_rst0, _rst3, _rst1, _rst2], axis = 0 )
            cv2.imshow( "wnd", rst )
            #cv2.imwrite('rst.jpg', 255*rst)    
            #cv2.waitKey(0)
            #cv2.imshow("aa", _rst0[0] )
            cv2.waitKey(1)

            movieimages1.append(rst[...,::-1])


        imageio.mimsave('movie2.gif', movieimages1)    


        """
        _depth_maps = 1.0/_rst2
        maxc = np.max( _depth_maps, axis=(1,2)).reshape(-1,1,1)
        tmp = _depth_maps
        tmp[ np.where(tmp==0) ] = np.float('inf')
        minc = np.min( tmp, axis=(1,2)).reshape(-1,1,1)

        _depth_maps = 0.1 + 0.9* (_depth_maps - minc)/(maxc-minc)
        """

        """

        batch_size = _landmarks.shape[0]
        num_landmarks = _landmarks.shape[1]

        """

    """
    ny = 1
    nx = batch_size
    height = width = IMAGE_WIDTH
    channels = 3
    _rst0 = _rst0.reshape(ny,nx,height,width,channels).transpose(0,2,1,3,4).reshape(height*ny,width*nx,channels)
    _rst1 = _rst1.reshape(ny,nx,height,width,channels).transpose(0,2,1,3,4).reshape(height*ny,width*nx,channels)
    _rst2 = _rst2.reshape(ny,nx,height,width).transpose(0,2,1,3).reshape(height*ny,width*nx)    
    _src0 = images_batch.reshape(ny,nx,height,width,channels).transpose(0,2,1,3,4).reshape(height*ny,width*nx,channels)    

    rst =  np.concatenate( [_src0,  _rst0, _rst1], axis = 0 )
    cv2.imshow( "wnd", rst )
    cv2.imwrite('rst2.jpg', 255*rst)    
    cv2.waitKey(0)
    """

    os._exit(0)

def rendering( filenames ):
    with open (uv_parsing_filename, 'rb') as fp:
        triangles, vertices_5d, landmarks = pickle.load(fp)
        assert NUM_VERTICES == vertices_5d.shape[0]

    uv_vertices_batch = vertices_5d[:,3:]
    uv_vertices_batch = np.expand_dims( uv_vertices_batch, axis = 0 )
    uv_vertices_batch = uv_vertices_batch * UV_PLANE_WIDTH // 2 
    uv_vertices_batch = tf.constant( uv_vertices_batch, dtype = tf.float32)

    batch_size = len(filenames)
    vertices_batch = []
    reflectances_batch = []
    images_batch = []

    for filename in filenames:
        imgfilename = filename.replace('.obj', '.jpg')
        img = cv2.imread( imgfilename )
        vertices, reflectances, trinagles  = load_obj(filename)
        vertices = vertices[:-4,:]
        reflectances = reflectances[:-4,:]
        print_ranges( vertices )

        vertices_batch.append( vertices )
        reflectances_batch.append( reflectances )
        images_batch.append( img )



    vertices_batch = np.array( vertices_batch )
    reflectances_batch = np.array( reflectances_batch )
    images_batch = np.array( images_batch ).astype(np.float32) / 255


    vertices = tf.constant( vertices_batch, dtype = tf.float32)
    reflectances = tf.constant( reflectances_batch, dtype = tf.float32)
    trinagles = tf.constant( trinagles[:-4], dtype = tf.int32 )


    vertices_indices = tf.constant( [[285],[1],[100]], dtype = tf.int32 )
    output_dictionary = dirt_utilities.rendering( None, vertices, reflectances, trinagles, vertices_indices )

    reflectances = ( - output_dictionary["vertex_normals"]  ) 
    reflectances = tf.nn.relu( reflectances )
    uv_vertices_batch = tf.tile( uv_vertices_batch, [batch_size, 1, 1 ])
    ortho_dictionary = dirt_utilities.dirt_rendering_orthographic( uv_vertices_batch, trinagles, reflectances = reflectances, \
                     frame_width = UV_PLANE_WIDTH, frame_height = -UV_PLANE_HEIGHT, cx = UV_PLANE_WIDTH//2, cy = UV_PLANE_HEIGHT/2  )
    

#    output_dictionary["depth_maps"] = tf.py_func( depth_normalize, \
#                [ output_dictionary["depth_maps"], STANDARD_DISTANCE - IMAGE_WIDTH//2, STANDARD_DISTANCE + IMAGE_WIDTH//2 ], tf.float64 )    

    with tf.Session() as sess:
        _rst0, _rst1, _rst2, _landmarks, _normals = sess.run( [output_dictionary["full_model_pixels"], output_dictionary["geometry_model_pixels"], \
                             output_dictionary["depth_maps"], output_dictionary["landmark_points"], ortho_dictionary["rendering_results"] ] )

        _rst0 = _rst0[:,::-1,::-1,:]
        _rst1 = _rst1[:,::-1,::-1,:]
        _rst2 = _rst2[:,::-1,::-1]    


        _normals = _normals[:,:,:,2:3]
        _normals = np.tile( _normals, [1,1,1,3])

        """
        _depth_maps = 1.0/_rst2
        maxc = np.max( _depth_maps, axis=(1,2)).reshape(-1,1,1)
        tmp = _depth_maps
        tmp[ np.where(tmp==0) ] = np.float('inf')
        minc = np.min( tmp, axis=(1,2)).reshape(-1,1,1)

        _depth_maps = 0.1 + 0.9* (_depth_maps - minc)/(maxc-minc)
        """

        batch_size = _landmarks.shape[0]
        num_landmarks = _landmarks.shape[1]
        for i in range(batch_size):
            for j in range(num_landmarks):
                x0 = int( _landmarks[i,j,0] )
                y0 = int( _landmarks[i,j,1] )

                x0 = IMAGE_WIDTH - x0
                y0 = IMAGE_WIDTH - y0

                img = _rst0[i].copy()
                cv2.circle( img, (x0,y0), 3, color=(0,1,1), thickness = -1 )
                _rst0[i] = img


    ny = 1
    nx = batch_size
    height = width = IMAGE_WIDTH
    channels = 3
    _rst0 = _rst0.reshape(ny,nx,height,width,channels).transpose(0,2,1,3,4).reshape(height*ny,width*nx,channels)
    _rst1 = _rst1.reshape(ny,nx,height,width,channels).transpose(0,2,1,3,4).reshape(height*ny,width*nx,channels)
    _rst2 = _rst2.reshape(ny,nx,height,width).transpose(0,2,1,3).reshape(height*ny,width*nx)    
    _normals = _normals.reshape(ny,nx,height,width,channels).transpose(0,2,1,3,4).reshape(height*ny,width*nx,channels)    
    _src0 = images_batch.reshape(ny,nx,height,width,channels).transpose(0,2,1,3,4).reshape(height*ny,width*nx,channels)    

    rst =  np.concatenate( [_src0,  _rst0, _rst1, _normals], axis = 0 )
    cv2.imshow( "wnd", rst )
    cv2.imwrite('rst2.jpg', 255*rst)    
    cv2.waitKey(0)



def perspective_face_example():
    with open (uv_parsing_filename, 'rb') as fp:
        triangles, vertices_5d, landmarks = pickle.load(fp)
        assert NUM_VERTICES == vertices_5d.shape[0]        

    vertices = vertices_5d[:,:3]

    filenames = ['./data_samples/obj_samples/0000.obj','./data_samples/obj_samples/0005.obj', \
        './data_samples/obj_samples/0010.obj', \
        './data_samples/obj_samples/0015.obj',  './data_samples/obj_samples/0020.obj','./data_samples/obj_samples/0025.obj', \
        './data_samples/obj_samples/0310.obj','./data_samples/obj_samples/0315.obj']

    #filenames = filenames[:2]
    assert len(filenames) % 2 == 0

    rendering( filenames )



def orthographic_example():
    z = 0
    vertices_batch = np.array( [ [[0,0], [100,0], [0,200], [-100,-100], [-200,200] ], [[0,0], [100,0], [0,200], [-100,-100], [-200,200] ] ]  , dtype = np.float32 )

    
    #reflectances_batch = np.array( [ [[1,0,0], [0,1,0], [0,0,1], [1,1,0] ], [[1,0,0], [0,1,0], [0,0,1], [1,1,0] ] ] , dtype = np.float32 )
    #reflectances_batch = np.ones_like( vertices_batch )
    triangles = [ [0,1,2], [0,1,3], [1,3,4] ] 
    triangles_np = np.array( triangles, dtype = np.int32 ) 

    print( vertices_batch.shape )

    vertices = tf.constant( vertices_batch, dtype = tf.float32)
    #reflectances = tf.constant( reflectances_batch, dtype = tf.float32)
    trinagles = tf.constant(triangles, dtype = tf.int32 )

    frame_width = 800
    frame_height = 800
    cx = 400
    cy = 400
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

def optimization_as_area():
    import imageio


    movieimages1 = []

    

    IMAGE_WIDTH = 512
    IMAGE_HEIGHT = 512
    NUM_POINTS = 60
    target_mask = np.zeros( [IMAGE_WIDTH, IMAGE_HEIGHT], dtype = np.float32 )
    cv2.circle( target_mask, (IMAGE_WIDTH//3, IMAGE_HEIGHT//2), radius=100, color=(1,1,1), thickness = -1 )
    cv2.circle( target_mask, (IMAGE_WIDTH*2//3, IMAGE_HEIGHT//2), radius=100, color=(1,1,1), thickness = -1 )
    cv2.circle( target_mask, (IMAGE_WIDTH//2, IMAGE_HEIGHT*2//3), radius=100, color=(1,1,1), thickness = -1 )
    target_mask_np = target_mask 


    triangles = []
    for i in range(1,NUM_POINTS-1):
        triangles.append( [0, i, i+1])
        #triangles.append( [i-1, i, i+1])
    triangles.append( [0,NUM_POINTS-1,1])

    target_mask = tf.constant( target_mask, dtype = tf.float32 )
    target_mask = tf.expand_dims( target_mask, axis = 0 )
    points = tf.get_variable( "points", shape=[1,NUM_POINTS-1,2], initializer=tf.initializers.truncated_normal(mean=0,stddev=30), dtype = tf.float32 )
    #points = np.zeros( [1,NUM_POINTS,2], dtype = np.float32 )
    #points = tf.Variable( points, dtype = tf.float32 )

    cx = IMAGE_WIDTH//2
    cy = IMAGE_HEIGHT//2



    bn_points = points 

    mean_points = tf.reduce_mean( bn_points, axis = 1)
    mean_points = tf.reshape( mean_points, [-1, 1, 2] )
    concatpoints = tf.concat( [ mean_points, bn_points ], axis=1)
    output_dictionary = dirt_utilities.dirt_rendering_orthographic( concatpoints, triangles, \
                                reflectances = None, frame_width = IMAGE_WIDTH, frame_height = IMAGE_HEIGHT, cx = cx, cy = cy  )

    rendering_result_3ch = output_dictionary["rendering_results"]
    rendering_result = rendering_result_3ch[:,:,:,0]

    loss = tf.reduce_sum( tf.abs( rendering_result  - target_mask) )
    loss = loss + tf.reduce_sum( tf.square( points[:,:-1,:] - points[:,1:,:])) * 0.1 +  tf.reduce_sum( tf.square( points[:,-1,:] - points[:,0,:])) * 0.1
    global_step = tf.Variable( 0, trainable=False)     

    learning_rate = 1

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

    update_var_list = tf.get_collection( key = tf.GraphKeys.TRAINABLE_VARIABLES )
    #print_var_status( update_var_list )

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss,global_step=global_step )

    font = cv2.FONT_HERSHEY_SIMPLEX
    yellow = (0,1,1)
    cyan = (1,1,0)


    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer() )
        i = 0
        while True:
            sess.run( train_op )
            i = i + 1
            if i % 100 == 0:
                _loss, _rendering_result, vertices_batch = sess.run( [loss, rendering_result_3ch, concatpoints])

                _rst0 = _rendering_result[0]
                for triangle in triangles:
                    u, v, w = triangle

                    pt1 = ( int(vertices_batch[0,u,0]  + cx) , int(-vertices_batch[0,u,1] + cy) )
                    pt2 = ( int(vertices_batch[0,v,0]  + cx) , int(-vertices_batch[0,v,1] + cy) )
                    pt3 = ( int(vertices_batch[0,w,0]  + cx) , int(-vertices_batch[0,w,1] + cy) )

                    cv2.putText(_rst0, str(u), pt1, font, .5, yellow, 1, cv2.LINE_AA)                           
                    cv2.putText(_rst0, str(v), pt2, font, .5, yellow, 1, cv2.LINE_AA)                           
                    cv2.putText(_rst0, str(w), pt3, font, .5, yellow, 1, cv2.LINE_AA)                           



                    cv2.line( _rst0, pt1, pt2, color=(1,0,0), thickness = 3 )
                    cv2.line( _rst0, pt2, pt3, color=(1,0,0), thickness = 3 )
                    cv2.line( _rst0, pt3, pt1, color=(1,0,0), thickness = 3 )




                rst = _rst0 *0.8 + np.expand_dims(target_mask_np,axis=-1) *0.2

                #cv2.imwrite( str(i) + ".jpg", 255*rst )
                cv2.imshow( "wnd", rst )
                movieimages1.append( rst )
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                print( _loss )


    imageio.mimsave('converge2.gif', movieimages1)

    cv2.imshow( "results", target_mask_np )
    cv2.waitKey(0)



if __name__ == "__main__":
    perspective_example()
    perspective_face_example()
    optimization_as_area()
    #orthographic_example()
    

    os._exit(0)


    with open (uv_parsing_filename, 'rb') as fp:
        triangles, vertices_5d, landmarks = pickle.load(fp)
        assert NUM_VERTICES == vertices_5d.shape[0]
        
    vertices = vertices_5d[:,:3]

    z = 0
    vertices_batch = vertices_5d[:,:3]

    vertices_batch = vertices_batch + 128
    vertices_batch[:,2] = vertices_batch[:,2] - STANDARD_DISTANCE 
    vertices_batch = vertices_batch / 255.
    vertices_batch = np.expand_dims( vertices_batch, axis = 0 )

    uv_vertices_batch = vertices_5d[:,3:]
    uv_vertices_batch = np.expand_dims( uv_vertices_batch, axis = 0 )

    frame_width = 256
    frame_height = 256
    cx = frame_width // 2
    cy = frame_height // 2
    uv_vertices_batch = uv_vertices_batch * frame_width // 2 
    
    triangles_np = triangles

    print( uv_vertices_batch.shape )

    vertices = tf.constant( uv_vertices_batch, dtype = tf.float32)
    trinagles = tf.constant(triangles, dtype = tf.int32 )

    reflectances = tf.constant( vertices_batch, dtype = tf.float32 )

    output_dictionary = dirt_utilities.dirt_rendering_orthographic( vertices, trinagles, reflectances = reflectances, frame_width = frame_width, frame_height = -frame_height, cx = cx, cy = cy  )


    font = cv2.FONT_HERSHEY_SIMPLEX
    yellow = (0,1,1)
    cyan = (1,1,0)    

    with tf.Session() as sess:
        _rst0, _rst1 = sess.run( [output_dictionary["rendering_results"], output_dictionary["vertices"] ] )
        _rst0 = _rst0[0]    

        """
        for j in range( triangles_np.shape[0] ):
            u, v, w = triangles_np[j]

            pt1 = ( int(vertices_batch[0,u,0]  + cx) , int(-vertices_batch[0,u,1] + cy) )
            pt2 = ( int(vertices_batch[0,v,0]  + cx) , int(-vertices_batch[0,v,1] + cy) )
            pt3 = ( int(vertices_batch[0,w,0]  + cx) , int(-vertices_batch[0,w,1] + cy) )
            
            cv2.line( _rst0, pt1, pt2, color=(1,0,0), thickness = 3 )
            cv2.line( _rst0, pt2, pt3, color=(1,0,0), thickness = 3 )
            cv2.line( _rst0, pt3, pt1, color=(1,0,0), thickness = 3 )
        """

        cv2.imshow( "rst0", _rst0 )
        cv2.waitKey(0)    