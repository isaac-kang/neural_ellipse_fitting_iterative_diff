import os,sys
sys.path.append( os.path.join(os.getcwd(), "../dirt") )

sys.path.append( os.path.join(os.getcwd(), "code_commons") )
from global_constants import * 

import dirt
import dirt.matrices as matrices
import dirt.lighting as lighting
import math
import numpy as np 
import tensorflow as tf 
from tensorflow.python.framework import ops

import cv2

def rendering( params_dictionary, vertices, reflectances, triangles, vertices_indices, frame_width = 256, frame_height = 256 ):
    batch_size = tf.shape( vertices )[0]  
    if params_dictionary is None:
        print("Default dictionary will be used")

        # use default 
        params_dictionary= {}
        params_dictionary["spherical_harmonics_parameters"] = tf.ones( [batch_size,27], dtype=tf.float32)
        params_dictionary["translation_parameters"] = tf.zeros( [batch_size,3], dtype=tf.float32)
        rotations = np.array( [[0,0,0]], dtype=np.float32)
        params_dictionary["rotation_parameters"] = tf.tile( tf.constant( rotations, dtype=tf.float32) , [batch_size, 1] )

        params_dictionary["focal_length"] = tf.ones( [batch_size], dtype=tf.float32) * NORMALIZED_FOCAL_LENGTH
        params_dictionary["principal_point_cx"] = tf.ones( [batch_size], dtype=tf.float32) * IMAGE_WIDTH // 2
        params_dictionary["principal_point_cy"] = tf.ones( [batch_size], dtype=tf.float32) * IMAGE_HEIGHT // 2

        rotations = np.array( [[np.pi,0,0]], dtype=np.float32)
        params_dictionary["camera_rotation_parameters"] = tf.tile( tf.constant( rotations, dtype=tf.float32) , [batch_size, 1] ) 


        rotations = np.array( [[0.00001,0,0]], dtype=np.float32)
        base_rotation = tf.tile( tf.constant( rotations, dtype=tf.float32) , [batch_size, 1] ) 



    else:
        if params_dictionary["convention"] == "option1":
            rotations = np.array( [[np.pi,0,0]], dtype=np.float32)
            base_rotation = tf.tile( tf.constant( rotations, dtype=tf.float32) , [batch_size, 1] ) 
            params_dictionary["spherical_harmonics_parameters"] = tf.ones( [batch_size,27], dtype=tf.float32)



    spherical_harmonics_parameters = params_dictionary["spherical_harmonics_parameters"] 
    translation_parameters= params_dictionary["translation_parameters"] 
    rotation_parameters = params_dictionary["rotation_parameters"] 
    camera_rotation_parameters = params_dictionary["camera_rotation_parameters"] 

    focal_length = params_dictionary["focal_length"]
    cx = frame_width - params_dictionary["principal_point_cx"]
    cy =  params_dictionary["principal_point_cy"]

    with tf.variable_scope("dirt_bazel"):
        with tf.variable_scope("basemodel"):
            output1_dict = dirt_rendering(  \
                        vertices, triangles, reflectances,
                            spherical_harmonics_parameters, translation_parameters, camera_rotation_parameters, base_rotation, rotation_parameters,\
                                frame_width, frame_height, \
                                    focal_length, cx, cy, vertices_indices  )

    return output1_dict





def get_landmarks( vertices, normals, indices ):
    landmarks_vertices = tf.transpose( vertices, [1,0,2])
    landmarks_normals = tf.transpose( normals, [1,0,2])

    landmarks_vertices = tf.gather_nd(landmarks_vertices, indices)
    landmarks_normals =  tf.gather_nd(landmarks_normals, indices)

    landmarks_vertices = tf.transpose( landmarks_vertices, [1,0,2])
    landmarks_normals = tf.transpose( landmarks_normals, [1,0,2])

    return landmarks_vertices, landmarks_normals


def spherical_harmonics( vertex_normals, vertex_refletances, sphereical_harmonic_coefficients, viewpoint_direction):
    vertex_normals = tf.convert_to_tensor(vertex_normals, name='vertex_normals')
    vertex_refletances = tf.convert_to_tensor(vertex_refletances, name='vertex_refletances')
    sphereical_harmonic_coefficients = tf.convert_to_tensor(sphereical_harmonic_coefficients, name='sphereical_harmonic_coefficients')
    sphereical_harmonic_coefficients = tf.reshape( sphereical_harmonic_coefficients, [-1,1,9,3])

    k0 = 0.5 * math.sqrt(1.0 / math.pi)
    k1 = math.sqrt(3.0) * k0
    k21 = math.sqrt(15.0) * k0
    k22 = 0.5 * math.sqrt(5.0) * k0
    
    nxs = vertex_normals[..., 0:1]  # shape: [batch_size, pixel_count, 1]
    nys = vertex_normals[..., 1:2]
    nzs = vertex_normals[..., 2:3]
    
    ones = tf.ones_like(nxs)
    nx_nys = tf.multiply(nxs, nys)
    ny_nzs = tf.multiply(nys, nzs)
    nz_nxs = tf.multiply(nzs, nxs)
    nx_sqrs = tf.multiply(nxs, nxs)
    ny_sqrs = tf.multiply(nys, nys)
    nz_sqrs = tf.multiply(nzs, nzs)
    sh_basis = tf.concat(  # shape: [batch_size, pixel_count, 9]
        [k0 * ones,
        k1 * nys, k1 * nzs, k1 * nxs,
        k21 * nx_nys, k21 * ny_nzs, k22 * (3.0 * nz_sqrs - 1.0), k21 * nz_nxs, k21 * (nx_sqrs - ny_sqrs)],
        axis=2)  
    sh_basis = tf.expand_dims(sh_basis, axis=-1)
    sh_basis_rgb = tf.concat([sh_basis, sh_basis, sh_basis], axis=-1)  # shape: [batch_size, pixel_count, 9, 3]
    
    vertex_colors = tf.multiply( vertex_refletances,  # shape: [batch_size, pixel_count, 3]
            tf.reduce_sum(tf.multiply( sh_basis_rgb, sphereical_harmonic_coefficients ), axis=(2) ) )
    
    vertex_colors = tf.clip_by_value(vertex_colors, 0.0, 1.0)  

    return vertex_colors

def _prepare_vertices_colors_and_faces(vertices, colors, faces):
    vertices = tf.convert_to_tensor(vertices, name='vertices')
    colors = tf.convert_to_tensor(colors, name='colors')
    faces = tf.convert_to_tensor(faces, name='faces')

    if faces.dtype is not tf.int32:
        assert faces.dtype is tf.int64
        faces = tf.cast(faces, tf.int32)

    return vertices, colors, faces


def _prepare_vertices_and_faces(vertices, faces):
    vertices = tf.convert_to_tensor(vertices, name='vertices')
    faces = tf.convert_to_tensor(faces, name='faces')

    if faces.dtype is not tf.int32:
        assert faces.dtype is tf.int64
        faces = tf.cast(faces, tf.int32)

    return vertices, faces


def split_vertices_by_color_normal_face(vertices, colors, normals, faces, name=None):
    """Returns a new mesh where each vertex is used by exactly one face.

    This function takes a batch of meshes with common topology as input, and also returns a batch of meshes
    with common topology. The resulting meshes have the same geometry, but each vertex is used by exactly
    one face.

    Args:
        vertices: a `Tensor` of shape [*, vertex count, 3] or [*, vertex count, 4], where * represents arbitrarily
            many leading (batch) dimensions.
        faces: an int32 `Tensor` of shape [face count, 3]; each value is an index into the first dimension of `vertices`, and
            each row defines one triangle.

    Returns:
        a tuple of two tensors `new_vertices, new_faces`, where `new_vertices` has shape [*, V, 3] or [*,  V, 4], where
        V is the new vertex count after splitting, and `new_faces` has shape [F, 3] where F is the new face count after
        splitting.
    """

    # This returns an equivalent mesh, with vertices duplicated such that there is exactly one vertex per face it is used in
    # vertices is indexed by *, vertex-index, x/y/z[/w]
    # faces is indexed by face-index, vertex-in-face
    # Ditto for results

    with ops.name_scope(name, 'SplitVerticesByFace', [vertices, colors, normals, faces]) as scope:

        vertices, colors, faces = _prepare_vertices_colors_and_faces(vertices, colors, faces)

        vertices_shape = tf.shape(vertices)
        face_count = tf.shape(faces)[0]

        flat_vertices = tf.reshape(vertices, [-1, vertices_shape[-2], vertices_shape[-1]])
        new_flat_vertices = tf.map_fn(lambda vertices_for_iib: tf.gather(vertices_for_iib, faces), flat_vertices)
        new_vertices = tf.reshape(new_flat_vertices, tf.concat([vertices_shape[:-2], [face_count * 3, vertices_shape[-1]]], axis=0))
        
        flat_colors = tf.reshape(colors, [-1, vertices_shape[-2], vertices_shape[-1]])
        new_flat_colors = tf.map_fn(lambda colors_for_iib: tf.gather(colors_for_iib, faces), flat_colors)
        new_colors = tf.reshape(new_flat_colors, tf.concat([vertices_shape[:-2], [face_count * 3, vertices_shape[-1]]], axis=0))
        
        flat_normals = tf.reshape(normals, [-1, vertices_shape[-2], vertices_shape[-1]])
        new_flat_normals = tf.map_fn(lambda normals_for_iib: tf.gather(normals_for_iib, faces), flat_normals)
        new_normals = tf.reshape(new_flat_normals, tf.concat([vertices_shape[:-2], [face_count * 3, vertices_shape[-1]]], axis=0))

        new_faces = tf.reshape(tf.range(face_count * 3), [-1, 3])

        static_face_count = faces.get_shape().dims[0] if faces.get_shape().dims is not None else None
        static_new_vertex_count = static_face_count * 3 if static_face_count is not None else None
        if vertices.get_shape().dims is not None:
            new_vertices.set_shape(vertices.get_shape().dims[:-2] + [static_new_vertex_count] + vertices.get_shape().dims[-1:])
            new_colors.set_shape(colors.get_shape().dims[:-2] + [static_new_vertex_count] + colors.get_shape().dims[-1:])
            new_normals.set_shape(normals.get_shape().dims[:-2] + [static_new_vertex_count] + normals.get_shape().dims[-1:])
  
        new_faces.set_shape([static_face_count, 3])
    

        return new_vertices, new_colors, new_normals, new_faces

def diffuse_directional_lights(vertex_normals, vertex_colors, light_direction, viewpoint_direction, light_color, double_sided=False, name=None):
    # vertex_normals is indexed by *, vertex-index, x/y/z; it is assumed to be normalised
    # vertex_colors is indexed by *, vertex-index, r/g/b
    # light_direction is indexed by *, x/y/z; it is assumed to be normalised
    # light_color is indexed by *, r/g/b
    # result is indexed by *, vertex-index, r/g/b

    with ops.name_scope(name, 'DiffuseDirectionalLight', [vertex_normals, vertex_colors, light_direction, light_color]) as scope:

        vertex_normals = tf.convert_to_tensor(vertex_normals, name='vertex_normals')
        vertex_colors = tf.convert_to_tensor(vertex_colors, name='vertex_colors')
        light_direction = tf.convert_to_tensor(light_direction, name='light_direction')
        light_color = tf.convert_to_tensor(light_color, name='light_color')

        cosines = tf.matmul(vertex_normals, -light_direction[..., tf.newaxis])  # indexed by *, vertex-index, singleton
        visibility = tf.matmul(vertex_normals, viewpoint_direction[..., tf.newaxis])

        if double_sided:
            cosines = tf.abs(cosines)
        else:
            cosines = tf.maximum(cosines, 0.)


        visibility = tf.minimum( visibility+0.9999, 1.0)
        visibility = tf.math.floor( visibility )
        return light_color[..., tf.newaxis, :] * vertex_colors  * cosines #* visibility




########################### Rendering using DIRT ##############################
def dirt_rendering( vertices, faces, reflectances, \
                    spherical_harmonics_parameters, translation_parameters, camera_rotation_parameters, base_rotation, rotation_parameters, \
                    frame_width, frame_height, focal_length, cx, cy, vertices_indices ):

    rendering_batch_size = tf.shape( vertices )[0]
    vertices = tf.matmul( vertices, matrices.rodrigues(rotation_parameters)[...,:3,:3])

    vertex_normals = normals = lighting.vertex_normals( vertices, faces )


    landmark_vertices, landmark_normals = get_landmarks( vertices, normals, vertices_indices )
    bz_facemodel_vertices_object, bz_facemodel_vertex_colors, bz_facemodel_vertex_normals, bz_facemodel_faces = split_vertices_by_color_normal_face(vertices, reflectances, normals, faces)

    
    # Transform vertices from world to camera space; note that the camera points along the negative-z axis in camera space
    view_matrix = matrices.compose(
#        matrices.rodrigues(rotation_parameters),
        matrices.translation(-translation_parameters),  # translate it away from the camera
        matrices.rodrigues(camera_rotation_parameters),        # tilt the view downwards
        matrices.rodrigues(base_rotation)
    )
    """

    view_matrix = matrices.compose(
        matrices.translation(translation_parameters),  # translate it away from the camera                
        matrices.rodrigues(camera_rotation_parameters),        # tilt the view downwards
        matrices.translation(-translation_parameters)  # translate it away from the camera        
    )
    """

    # Convert vertices to homogeneous coordinates
    bz_facemodel_vertices_object = tf.concat([
        bz_facemodel_vertices_object,
        tf.ones_like(bz_facemodel_vertices_object[..., -1:])
    ], axis=-1)

    landmark_vertices = tf.concat([
        landmark_vertices,
        tf.ones_like(landmark_vertices[..., -1:])
    ], axis=-1)

    """
    with tf.Session() as sess:
        _view_matrix = sess.run( view_matrix )
        print( _view_matrix )
    """



    viewpoint_direction = -tf.ones_like( translation_parameters )
    norms = tf.norm(viewpoint_direction, axis=-1, keep_dims=True)  # indexed by *, singleton
    viewpoint_direction /= (norms+1e-5)


    viewpoint_direction = np.array( [0,0,1], np.float32 )
    viewpoint_direction = np.expand_dims( viewpoint_direction, axis=0 )
    viewpoint_direction = tf.constant( viewpoint_direction )
    viewpoint_direction = tf.tile( viewpoint_direction, [rendering_batch_size,1])


    # Transform vertices from object to world space, by rotating around the vertical axis
    bz_facemodel_vertices_world = bz_facemodel_vertices_object     #  tf.matmul(cube_vertices_object, [matrices.rodrigues([0.0, 0.0, 0.0])] )
    landmark_vertices_world = landmark_vertices


    # Calculate face normals; pre_split implies that no faces share a vertex
    bz_facemodel_faces = tf.expand_dims( bz_facemodel_faces, axis = 0 )                         
    bz_facemodel_faces = tf.tile( bz_facemodel_faces, (rendering_batch_size,1,1) )              

    bz_facemodel_normals_world = bz_facemodel_vertex_normals       #  lighting.vertex_normals_pre_split(cube_vertices_world, cube_faces)
    landmark_vertices_normals = landmark_normals

    bz_facemodel_vertices_camera = tf.matmul(bz_facemodel_vertices_world, view_matrix )
    landmark_vertices_camera = tf.matmul(landmark_vertices_world, view_matrix )


    # Transform vertices from camera to clip space
    near = focal_length * 0.1           # 0.01 just constant 
    far = focal_length * 100 
#    right = frame_width * 0.5 * 0.01
#    projection_matrix = matrices.perspective_projection(near=near, far=1000., right=right, aspect=float(frame_height) / frame_width)
    projection_matrix = matrices.perspective_projection( near=near, far=far, fx = focal_length, fy=focal_length, \
        w = frame_width, h = frame_height, cx = cx, cy = cy )

   



#    projection_matrix = tf.expand_dims( projection_matrix, axis = 0)
#    projection_matrix = tf.tile( projection_matrix, (rendering_batch_size,1,1))

    bz_facemodel_vertices_clip = tf.matmul(bz_facemodel_vertices_camera, projection_matrix )
    landmark_vertices_vertices_clip = tf.matmul(landmark_vertices_camera, projection_matrix )




    
    #K = tf.constant( 
    #    np.array( [ [focal_length,0,frame_width*0.5],[0,focal_length,frame_height*0.5],[0.0,0.0,1.0]] ), dtype = tf.float32) 

    """
    ### if you want check projection matrix 
    with tf.Session() as sess:
        _landmarks = sess.run( landmark_vertices )    
        _view_matrix = sess.run( view_matrix )
        _projection_matrix = sess.run( projection_matrix )
        #_K = sess.run(K )
        _T = sess.run( matrices.translation(translation_parameters) )
        _R = sess.run( matrices.rodrigues(rotation_parameters)  )

        print(np.array2string(_T, separator=', ')) 
        print(np.array2string(_R, separator=', ')) 
        print(np.array2string(_landmarks, separator=', ')) 
        print(np.array2string(_view_matrix, separator=', ')) 
        #print(np.array2string(_K, separator=', ')) 
        print(np.array2string(_projection_matrix, separator=', ')) 
    """

    """
    # Calculate lighting, as combination of diffuse and ambient
    vertex_colors_lit = diffuse_directional_lights(
        bz_facemodel_normals_world, bz_facemodel_vertex_colors,
        light_direction=light_directions, viewpoint_direction=viewpoint_direction, light_color=light_colors
    )  * 1.0 #+ bz_facemodel_vertex_colors * 0.2
    """


    # geometry 
    use_spherical_harmonics = False


    if use_spherical_harmonics == True:
        vertex_colors_lit = spherical_harmonics( bz_facemodel_normals_world, bz_facemodel_vertex_colors, spherical_harmonics_parameters, viewpoint_direction=viewpoint_direction )

        geometry_visualization_spherical_harmonics_parameters = np.zeros([27], dtype=np.float32)
        geometry_visualization_spherical_harmonics_parameters[3:3+9] = 1.0
        geometry_visualization_spherical_harmonics_parameters = tf.constant( geometry_visualization_spherical_harmonics_parameters, dtype = tf.float32 )
        geometry_visualization_vertex_colors_lit = spherical_harmonics( bz_facemodel_normals_world, tf.ones_like(bz_facemodel_vertex_colors), \
                                geometry_visualization_spherical_harmonics_parameters, viewpoint_direction=viewpoint_direction )
    else:
        light_directions = np.array( [
            [0,0,1]
        ])
        norm = np.linalg.norm(light_directions,axis=-1)
        light_directions = light_directions / norm[:,np.newaxis]
        light_directions = tf.constant( light_directions, dtype = tf.float32 )
        light_directions = tf.tile( light_directions, (rendering_batch_size,1))

        light_colors = tf.constant(  [1., 1., 1.], dtype=tf.float32 )
        light_colors = tf.expand_dims( light_colors, axis = 0)
        light_colors = tf.tile( light_colors, (rendering_batch_size,1))

        geometry_visualization_vertex_colors_lit = diffuse_directional_lights( bz_facemodel_normals_world,  tf.ones_like(bz_facemodel_vertex_colors), \
                                    light_direction=light_directions, viewpoint_direction=viewpoint_direction, light_color=light_colors, double_sided = False ) * 1.0 #+ bz_facemodel_vertex_colors * 0.2
        vertex_colors_lit = diffuse_directional_lights( bz_facemodel_normals_world,  bz_facemodel_vertex_colors, light_direction=light_directions, \
                                            viewpoint_direction=viewpoint_direction, light_color=light_colors ) * 1.0 #+ bz_facemodel_vertex_colors * 0.2

    # reflectance 
    reflectance_visualization_spherical_harmonics_parameters = np.zeros([27], dtype=np.float32)
    reflectance_visualization_spherical_harmonics_parameters[0:3] = 3.0
    reflectance_visualization_spherical_harmonics_parameters = tf.constant( reflectance_visualization_spherical_harmonics_parameters, dtype = tf.float32 )
    reflectance_visualization_vertex_colors_lit = spherical_harmonics( bz_facemodel_normals_world, bz_facemodel_vertex_colors, \
                                reflectance_visualization_spherical_harmonics_parameters, viewpoint_direction=viewpoint_direction )

    # illumination 
    illumination_visualization_vertex_colors_lit = spherical_harmonics( bz_facemodel_normals_world, tf.ones_like(bz_facemodel_vertex_colors), \
                            spherical_harmonics_parameters, viewpoint_direction=viewpoint_direction )


    # depth 
    depth_vertex_colors_lit = tf.concat( 
        [ geometry_visualization_vertex_colors_lit[:,:,0:1], bz_facemodel_normals_world[:,:,2:3], bz_facemodel_vertices_camera[:,:,2:3] ], axis = -1 )



    # landmarks 
    denom = landmark_vertices_vertices_clip[:,:,3]
    landmark_points = landmark_vertices_vertices_clip[:,:,0:2]/(denom[...,tf.newaxis])
    landmark_xpoint = ( 1.0 + landmark_points[:,:,0:1] ) * frame_width * 0.5
    landmark_ypoint = ( 1.0 - landmark_points[:,:,1:2]) * frame_height * 0.5
    landmark_points = tf.concat( [landmark_xpoint, landmark_ypoint], axis=-1)

    """
    with tf.Session() as sess:
        _AAA = sess.run( landmark_points )
        print(_AAA)
    """

    # landmark visibility 
    landmark_visibility = tf.matmul(landmark_vertices_normals, viewpoint_direction[..., tf.newaxis])
    #visibility = tf.minimum( landmark_visibility+0.9999, 1.0)
    #visibility = tf.cast( visibility, tf.int32 )


    full_model_pixels = dirt.rasterise_batch(
        vertices=bz_facemodel_vertices_clip,
        faces=bz_facemodel_faces,
        vertex_colors=vertex_colors_lit,
        background=tf.zeros([rendering_batch_size,frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )

    geometry_model_pixels = dirt.rasterise_batch(
        vertices=bz_facemodel_vertices_clip,
        faces=bz_facemodel_faces,
        vertex_colors=geometry_visualization_vertex_colors_lit,
        background=tf.zeros([rendering_batch_size,frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )

    reflectance_model_pixels = dirt.rasterise_batch(
        vertices=bz_facemodel_vertices_clip,
        faces=bz_facemodel_faces,
        vertex_colors=tf.ones_like(reflectance_visualization_vertex_colors_lit),
        background=tf.zeros([rendering_batch_size,frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )

    illumination_model_pixels = dirt.rasterise_batch(
        vertices=bz_facemodel_vertices_clip,
        faces=bz_facemodel_faces,
        vertex_colors=illumination_visualization_vertex_colors_lit,
        background=tf.zeros([rendering_batch_size,frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )


    depth_maps = - dirt.rasterise_batch(
        vertices=bz_facemodel_vertices_clip,
        faces=bz_facemodel_faces,
        vertex_colors=depth_vertex_colors_lit,  
        background=tf.zeros([rendering_batch_size,frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )


    output1_dict = {}
    output1_dict["full_model_pixels"] = full_model_pixels
    output1_dict["vertices_clip"] = bz_facemodel_vertices_clip
    output1_dict["landmark_points"] = landmark_points
    output1_dict["landmark_visibility"] = landmark_visibility
    #output1_dict["depth_masks"] =  tf.clip_by_value( 5*( -depth_maps[:,:,:,0] ), 0, 1 ) # tf.stop_gradient( ???
    #output1_dict["depth_masks"] =   tf.stop_gradient( tf.clip_by_value( 100*( -depth_maps[:,:,:,0] ), 0, 1 ) )
    output1_dict["depth_maps"] = depth_maps[:,:,:,2]
    output1_dict["geometry_model_pixels"] = geometry_model_pixels
    output1_dict["reflectance_model_pixels"] = reflectance_model_pixels
    output1_dict["illumination_model_pixels"] = illumination_model_pixels
    output1_dict["surface_normals"] = (1+depth_maps[:,:,:,1])/2.0
    output1_dict["vertex_normals"] = vertex_normals


    return output1_dict





def orthographic_projection(near, far, ones, w, h, cx ,cy, name=None):
    """Constructs a perspective projection matrix.

    This function returns a perspective projection matrix, using the OpenGL convention that the camera
    looks along the negative-z axis in view/camera space, and the positive-z axis in clip space.
    Multiplying view-space homogeneous coordinates by this matrix maps them into clip space.

    Args:
        near: distance to the near clipping plane; geometry nearer to the camera than this will not be rendered
        far: distance to the far clipping plane; geometry further from the camera than this will not be rendered
        right: distance of the right-hand edge of the view frustum from its centre at the near clipping plane
        aspect: aspect ratio (height / width) of the viewport
        name: an optional name for the operation

    Returns:
        a 4x4 `Tensor` containing the projection matrix
    """

    with ops.name_scope(name, 'OrthogrhicProjection', [near, far, w, h, cx, cy]) as scope:
        near = tf.convert_to_tensor(near, name='near')
        far = tf.convert_to_tensor(far, name='far')

        cx = tf.convert_to_tensor(cx, name='cx')
        cy = tf.convert_to_tensor(cy, name='cy')

        w = w * ones
        h = h * ones

        zeros = tf.zeros_like( ones )
    
        
        elements = [
            [2.0*ones/w,zeros,zeros,2.0*(cx/w)-ones],
            [zeros,2.0*ones/h,zeros,2.0*(cy/h)-ones],
            [zeros, zeros,-2./(far-near),(far+near)/(far-near)],
            [zeros, zeros, zeros, ones ]     
        ]  # indexed by x/y/z/w (out), x/y/z/w (in)
        return tf.transpose(tf.convert_to_tensor(elements, dtype=tf.float32))


"""
    with ops.name_scope(name, 'PerspectiveProjection', [near, far, fx, fy, w, h, cx, cy]) as scope:
        near = tf.convert_to_tensor(near, name='near')
        far = tf.convert_to_tensor(far, name='far')

        fx = tf.convert_to_tensor(fx, name='fx')
        fy = tf.convert_to_tensor(fy, name='fy')

        cx = tf.convert_to_tensor(cx, name='cx')
        cy = tf.convert_to_tensor(cy, name='cy')

        w = w * tf.ones_like( fx) 
        h = h * tf.ones_like( fy) 

        zeros = tf.zeros_like( fx )
        ones = tf.ones_like( fx )
        
        elements = [
            [2.0*fx/w,zeros,2.0*(cx/w)-ones,zeros],
            [zeros,2.0*fy/h,2.0*(cy/h)-ones,zeros],
            [zeros,zeros,-(far+near)/(far-near),-2*far*near/(far-near)],
            [zeros, zeros, -ones, zeros]     
        ]  # indexed by x/y/z/w (out), x/y/z/w (in)
        return tf.transpose(tf.convert_to_tensor(elements, dtype=tf.float32))
"""



def dirt_rendering_orthographic( vertices, faces, reflectances, \
                    frame_width, frame_height, cx, cy, vertices_to_keep_track = None ):

    if vertices.shape[-1] == 2:
        vertices = tf.concat([
            vertices,
            tf.zeros_like(vertices[..., -1:])
        ], axis=-1)

    vertical_flip = False 
    if frame_height < 0:
        frame_height = - frame_height
        vertical_flip = True



    if reflectances is None:
        reflectances = tf.ones_like( vertices)

    rendering_batch_size = tf.shape( vertices )[0]
    ones = tf.ones( [rendering_batch_size], dtype=tf.float32)
    normals = lighting.vertex_normals( vertices, faces )

    bz_facemodel_vertices_object, bz_facemodel_vertex_colors, bz_facemodel_vertex_normals, bz_facemodel_faces = split_vertices_by_color_normal_face(vertices, reflectances, normals, faces)

    if vertices_to_keep_track is None:
        landmark_vertices = vertices
        landmark_normals = normals
    else:
        landmark_vertices, landmark_normals = get_landmarks( vertices, normals, vertices_to_keep_track )


    translation_parameters = np.array( [[0,0,100]], dtype=np.float32)
    translation_parameters= tf.tile( tf.constant( translation_parameters, dtype=tf.float32) , [rendering_batch_size, 1] )
    rotation_parameters = np.array( [[1e-10,0,0]], dtype=np.float32)
    rotation_parameters= tf.tile( tf.constant( rotation_parameters, dtype=tf.float32) , [rendering_batch_size, 1] )
    cx = tf.ones( [rendering_batch_size], dtype=tf.float32) * cx
    cy = tf.ones( [rendering_batch_size], dtype=tf.float32) * (frame_height - cy)
    


    # Transform vertices from world to camera space; note that the camera points along the negative-z axis in camera space
    view_matrix = matrices.compose(
        matrices.translation(translation_parameters),  # translate it away from the camera
        matrices.rodrigues(rotation_parameters)        # tilt the view downwards
    )

    """
    with tf.Session() as sess: 
        _batch = sess.run( rendering_batch_size )
        _a = sess.run( matrices.translation(translation_parameters) )
        _b = sess.run( matrices.rodrigues(rotation_parameters)  )
        _c = sess.run( view_matrix )

        print("translation", np.transpose(_a[0]))
        print("rotation",  np.transpose(_b[0]))
        print("camera",  np.transpose(_c[0]))
    """


    # Convert vertices to homogeneous coordinates
    bz_facemodel_vertices_object = tf.concat([
        bz_facemodel_vertices_object,
        tf.ones_like(bz_facemodel_vertices_object[..., -1:])
    ], axis=-1)

    landmark_vertices = tf.concat([
        landmark_vertices,
        tf.ones_like(landmark_vertices[..., -1:])
    ], axis=-1)    

    viewpoint_direction = - tf.ones_like( translation_parameters )
    norms = tf.norm(viewpoint_direction, axis=-1, keep_dims=True)  # indexed by *, singleton
    viewpoint_direction /= norms

    viewpoint_direction = np.array( [0,0,-1], np.float32 )
    viewpoint_direction = np.expand_dims( viewpoint_direction, axis=0 )
    viewpoint_direction = tf.constant( viewpoint_direction )
    viewpoint_direction = tf.tile( viewpoint_direction, [rendering_batch_size,1])


    # Transform vertices from object to world space, by rotating around the vertical axis
    bz_facemodel_vertices_world = bz_facemodel_vertices_object     #  tf.matmul(cube_vertices_object, [matrices.rodrigues([0.0, 0.0, 0.0])] )
    landmark_vertices_world = landmark_vertices

    # Calculate face normals; pre_split implies that no faces share a vertex
    bz_facemodel_faces = tf.expand_dims( bz_facemodel_faces, axis = 0 )                         
    bz_facemodel_faces = tf.tile( bz_facemodel_faces, (rendering_batch_size,1,1) )              
    bz_facemodel_normals_world = bz_facemodel_vertex_normals       #  lighting.vertex_normals_pre_split(cube_vertices_world, cube_faces)
    bz_facemodel_vertices_camera = tf.matmul(bz_facemodel_vertices_world, view_matrix )


    # Transform vertices from camera to clip space
    near = 10.0 * ones 
    far = 200.0 * ones 

    projection_matrix = orthographic_projection( near=near, far=far, w = frame_width, ones = ones, h = frame_height, cx = cx, cy = cy )
    bz_facemodel_vertices_clip = tf.matmul(bz_facemodel_vertices_camera, projection_matrix )
    landmark_vertices_vertices_clip = tf.matmul(landmark_vertices_world, projection_matrix )

    """
    with tf.Session() as sess: 
        
        _v0 = sess.run( bz_facemodel_vertices_world )
        _v1 = sess.run( bz_facemodel_vertices_camera )
        _v2 = sess.run( bz_facemodel_vertices_clip )


        print("vertices\n", (_v0[0]))
        print("vertices\n", (_v1[0]))
        print("vertices\n", (_v2[0]))
    """

    full_model_pixels = dirt.rasterise_batch(
        vertices=bz_facemodel_vertices_clip,
        faces=bz_facemodel_faces,
        vertex_colors=bz_facemodel_vertex_colors,
        background=tf.zeros([rendering_batch_size,frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )



    # landmarks 
    denom = landmark_vertices_vertices_clip[:,:,3]
    landmark_points = landmark_vertices_vertices_clip[:,:,0:2]/(denom[...,tf.newaxis])
    landmark_xpoint = ( 1.0 + landmark_points[:,:,0:1] ) * frame_width * 0.5
    landmark_ypoint = ( 1.0 + landmark_points[:,:,1:2]) * frame_height * 0.5
    landmark_points = tf.concat( [landmark_xpoint, landmark_ypoint], axis=-1)


    output_dictionary = {}
    output_dictionary["rendering_results"] = full_model_pixels
    output_dictionary["vertices"] = landmark_points
    output_dictionary["landmark_normals"] = landmark_normals

    """
    with tf.Session() as sess:
        _vertices = sess.run( bz_facemodel_vertices_clip )
        print( _vertices.shape )
        print( _vertices )
    """

    if vertical_flip is True:
        output_dictionary["rendering_results"] = output_dictionary["rendering_results"][:,::-1,:,:]    

    return output_dictionary




if __name__ == "__main__":

    orthographic_example()
