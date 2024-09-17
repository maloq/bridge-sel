Suppose we have this data:
- the data of the image/pose DJI_20240418185815_0195_Z
- the data of the camera d0c8b88a-d4f1-4637-ac4f-85da04e7bb40 (sensor id 0) related to the pose
    camera_info = {
        "id": "d0c8b88a-d4f1-4637-ac4f-85da04e7bb40",
        "offset_x": 8.309657515284046,
        "offset_y": 1.8107673354504856,
        "focal_length": 4800.232015738299,
        "k1": -0.008434033956924982,
        "k2": 0.07688060637564556,
        "k3": -0.07907662055042962,
        "p1": 0.0009069205914153655,
        "p2": 1.402695059969209e-05,
        "width": 5184,
        "height": 3888,
        "matrix": [
            [
                4800.232015738299,
                0.0,
                2600.309657515284
            ],
            [
                0.0,
                4800.232015738299,
                1945.8107673354505
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ],
        "distortion_coefficients": [
            -0.008434033956924982,
            0.07688060637564556,
            0.0009069205914153655,
            1.402695059969209e-05,
            -0.07907662055042962
        ]
    }

    pose_info = {
        "center_x": -21.72083936294075,
        "center_y": -34.34368678275496,
        "center_z": 2.7943027446324322,
        "rotation_00": 0.9694059114998105,
        "rotation_01": -0.2451408350424042,
        "rotation_02": 0.0125757601732269,
        "rotation_10": 0.0499116335980566,
        "rotation_11": 0.1466943041991313,
        "rotation_12": -0.9879218642924684,
        "rotation_20": 0.2403351983809179,
        "rotation_21": 0.9583249720790156,
        "rotation_22": 0.1544417052125467,
        "image_original_name": "DJI_20240418185815_0195_Z",
        "center": [
            -21.72083936294075,
            -34.34368678275496,
            2.7943027446324322
        ],
        "rotation": [
            [
                0.9694059114998105,
                -0.2451408350424042,
                0.0125757601732269
            ],
            [
                0.0499116335980566,
                0.1466943041991313,
                -0.9879218642924684
            ],
            [
                0.2403351983809179,
                0.9583249720790156,
                0.1544417052125467
            ]
        ],
        "camera_id": "d0c8b88a-d4f1-4637-ac4f-85da04e7bb40"
    }

As you did with slice_mesh_with_fuse you can get the mesh seen by the camera
    camera_matrix = np.float32(camera_info["matrix"])
    distortion_coefficients = np.float32(camera_info["distortion_coefficients"])
    rotation = np.float32(pose_info["rotation"]).reshape(3, 3)
    center = np.float32(pose_info["center"]).reshape(3, 1)

    pier = trimesh.load(MESH_PATH, force='mesh')
    pier_cutted = slice_mesh_with_fuse(rotation, center, camera_matrix, camera_info["height"], camera_info["width"], pier)

    is_mesh_seen_by_camera = len(pier_cutted.vertices) != 0


You can get the projected vertices with cv.projectPoints (you should use the rotation matrix instead of vector)

    vertices_camera = world_to_camera(pier_cutted.vertices, rotation, center)
    translation_cam = (-rotation @ center).reshape(3, 1)
    projected_vertices, _ = cv.projectPoints(vertices_camera, rotation, translation_cam, camera_matrix, distortion_coefficients)
    projected_vertices = projected_vertices.squeeze().astype(np.int64)

You can use ray tracing to understand if vertices are visible or occluded

    rays_directions = projected_vertices - center
    distances = np.linalg.norm(rays_directions, axis=1)
    rays_directions /= distances.reshape(-1, 1)
    rays_origins = np.tile(center, (len(rays_directions), 1))

    locations, index_ray, index_tri = pier_cutted.ray.intersects_location(
        ray_origins=rays_origins,
        ray_directions=rays_directions
    )

    if locations.size == 0:
        print("error")
        
    # Calculate distances from the center of all the projected points
    distances = np.linalg.norm(locations - center.transpose(), axis=1).reshape(-1, 1)
    
    defect_3d_vertices = np.zeros((projected_vertices, 3), dtype=np.float32)
        correctly_projected_points = 0
        for i in range(projected_vertices):
        projected_points_for_i = locations[index_ray == i]
        # Validate that the projection of the single point has a valid result
        if projected_points_for_i.size == 0:
           break

        distances_ray_i = distances[index_ray == i]
        closest_index = np.argmin(distances_ray_i)
        defect_3d_vertices[i, :] = projected_points_for_i[closest_index]
        correctly_projected_points += 1

    
The approach should be

Ray tracing →draw a ray from the image’s centroid to each vertex.

Intersection check →determine if the nearest intersection of the ray with the mesh is at the selected vertex. If it is, the vertex is visible from the camera pose. If another intersection occurs closer to the camera, the vertex is occluded.






