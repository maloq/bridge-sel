def visible_points3(mesh, potential_indices, R, t,vertices, image_shape, camera_matrix):

    scene = mesh.scene()
    scene.set_camera(angles=rotation_matrix_to_euler_angles(R), distance=5, center=t)

    scene.camera.resolution = image_shape[::-1]
    fov_x = np.rad2deg(2 * np.arctan2(image_shape[1], 2 * camera_matrix[0,0]))
    fov_y = np.rad2deg(2 * np.arctan2(image_shape[0], 2 * camera_matrix[1,1]))
    print('Camera FOV', (fov_x, fov_y))
    scene.camera.focal = (camera_matrix[0,0], camera_matrix[1,1])
    origins, vectors, pixels = scene.camera_rays()
 

    # do the actual ray- mesh queries
    points, index_ray, index_tri = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False
    )

    # for each hit, find the distance along its vector
    depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])
    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]

    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)

    # scale depth against range (0.0 - 1.0)
    depth_float = (depth - depth.min()) / np.ptp(depth)

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)
    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
    # create a PIL image from the depth queries
    img = PIL.Image.fromarray(a)
    return img
    # show the resulting image
    #img.show()


def visible_points2(mesh, potential_indices, camera_position, vertices, image_shape, camera_matrix):

    ray_origins = np.tile(camera_position, (len(potential_indices), 1))
    ray_directions = vertices[potential_indices] - camera_position
    ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]

    intersector = ray.RayMeshIntersector(mesh)
    ray_idxs = intersector.intersects_id(ray_origins, ray_directions, multiple_hits=False)[1]
    visible_indices =  potential_indices[ray_idxs]
    img = np.zeros(image_shape[::-1])
    img[visible_indices] = 255
    img = PIL.Image.fromarray(img)
    return img
