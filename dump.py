def visible_points3(mesh, potential_indices, R, t,vertices, image_shape, camera_matrix):

    scene = mesh.scene()
    scene.set_camera(angles=rotation_matrix_to_euler_angles(R), distance=5, center=t)

    scene.camera.resolution = image_shape[::-1]
    fov_x = np.rad2deg(2 * np.arctan2(image_shape[1], 2 * camera_matrix[0,0]))
    fov_y = np.rad2deg(2 * np.arctan2(image_shape[0], 2 * camera_matrix[1,1]))
    print('Camera FOV', (fov_x, fov_y))
    scene.camera.focal = (camera_matrix[0,0], camera_matrix[1,1])
    origins, vectors, pixels = scene.camera_rays()
 

    points, index_ray, index_tri = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=False
    )

    depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])
    pixel_ray = pixels[index_ray]

    a = np.zeros(scene.camera.resolution, dtype=np.uint8)

    depth_float = (depth - depth.min()) / np.ptp(depth)

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
    img = PIL.Image.fromarray(a)
    return img
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



def manual_project(point, rotation, translation, camera_matrix):
    point = np.array(point).reshape(3, 1)
    translation = np.array(translation).reshape(3, 1)
    if rotation.shape == (3, 3):
        R = rotation
    elif rotation.shape == (3, 1) or rotation.shape == (1, 3):
        R, _ = cv.Rodrigues(rotation)
    else:
        raise ValueError("Rotation must be a 3x3 matrix or a 3x1 vector")
    point_camera = R @ point + translation
    x, y, z = point_camera.flatten()
    u = camera_matrix[0,0] * x/z + camera_matrix[0,2]
    v = camera_matrix[1,1] * y/z + camera_matrix[1,2]
    return np.array([u, v])

print("vertices_camera shape:", vertices_camera.shape)
print("rotation shape:", rotation.shape)
print("translation_cam shape:", translation_cam.shape)
print("camera_matrix shape:", camera_matrix.shape)

for i in range(5):
    point = vertices_camera[i]
    projected = manual_project(point, rotation, translation_cam, camera_matrix)
    print(f"Point {i}: 3D = {point}, Projected = {projected}")


image_shape = (5460, 8192)

camera_matrix = np.float32(camera_info["matrix"])
distortion_coefficients = np.float32(camera_info["distortion_coefficients"])
rotation = np.float32(pose_info["rotation"]).reshape(3, 3)
center = np.float32(pose_info["center"]).reshape(3, 1)
pose_vertices = pier_cutted.vertices.astype(np.float32)
translation_cam = (-rotation @ center).reshape(3, 1)

print(f"Total vertices: {len(pose_vertices)}")

rotation_vector, _ = cv.Rodrigues(rotation)
coords = pier_cutted.vertices.view(np.ndarray).astype(np.float32)

assert pose_vertices.dtype == np.float32
assert rotation_vector.dtype == np.float32
assert translation_cam.dtype == np.float32
assert camera_matrix.dtype == np.float32
assert distortion_coefficients.dtype == np.float32

# Proietta i vertici nello spazio 2D
projected_vertices, _ = cv.projectPoints(coords, rotation_vector, translation_cam, camera_matrix, distortion_coefficients)
projected_vertices = projected_vertices.squeeze().astype(np.int64)

def visualize_projected(vertices_image, image_size):
    plt.figure(figsize=(10, 10))
    plt.scatter(vertices_image[:, 0], vertices_image[:, 1], c='blue', s=1, alpha=0.5, label='All vertices')
    plt.xlim(0, image_size[1])
    plt.ylim(image_size[0], 0)  # Invert y-axis to match image coordinates
    plt.legend()
    plt.title('Projected Vertices')
    plt.show()


visualize_projected(projected_vertices, image_shape)