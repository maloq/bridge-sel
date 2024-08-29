import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import trimesh
import json

def slice_mesh_with_fuse(rotation, center, camera_matrix, image_width, image_height, mesh):
    """
    @brief Slice a mesh according to the fuse of a camera
    @param rotation Camera rotation matrix
    @param center   Camera center in world coordinates
    @param mesh     Target mesh to slice
    @return         A new mesh with only the portion inside the camera fuse
    """

    v = mesh.vertices.copy()
    f = mesh.faces.copy()

    # Extreme pixels in homogenous coordinates
    extremes = np.float32([
        [0.0, image_width, image_width, 0.0],
        [0.0, 0.0, image_height, image_height],
        [1.0, 1.0, 1.0, 1.0]
    ])

    camera_matrix_inv = np.linalg.inv(camera_matrix)
    extremes_3d = np.matmul(camera_matrix_inv, extremes)

    for j in range(extremes_3d.shape[1]):
        point0 = extremes_3d[:, j - 1]
        point1 = extremes_3d[:, j]
        point2 = np.float32([0, 0, 0])
        normal_to_camera = np.cross(point2 - point1, point0 - point1)
        normal_to_world = np.matmul(rotation.transpose(), normal_to_camera).flatten()
        v, f, _ = trimesh.intersections.slice_faces_plane(v, f, normal_to_world, center.flatten())
        
    return trimesh.Trimesh(vertices=v, faces=f)


def is_fully_visible(object_points, center, target_mesh, tol=1.0e-8):
    # Calculate rays to the object points
    ray_directions = object_points - center.transpose()
    # Project on the bridge
    points_count = object_points.shape[0]
    ray_origins = np.tile(center, (points_count,)).transpose()
    # TODO use index_tri to guess what is in the middle
    locations, index_ray, _ = target_mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions)
    # Pick the closest point of each ray
    location_distances = np.linalg.norm(locations - center.transpose(), axis=1).reshape(-1, 1)
    closest_locations = np.zeros((points_count, 3))
    for i in range(points_count):
        closest_locations[i, :] = locations[index_ray == i][np.argmin(location_distances[index_ray == i])]
    # If the closest locations are == to the initial object_points, then it is visible
    diff = closest_locations - object_points
    dist = np.linalg.norm(diff.ravel())
    return dist < tol


def crop(origin_u:int, origin_v:int, width:int, height:int, img):
    # Sanity checks
    if origin_u < 0:
        raise Exception("Invalid origin u")
    if origin_v < 0:
        raise Exception("Invalid origin v")
    if (origin_u + width) > img.shape[1]:
        raise Exception("Invalid cropping width")
    if (origin_v + height) > img.shape[0]:
        raise Exception("Invalid cropping height")
    
    return img[origin_v:(origin_v + height), origin_u:(origin_u + width)]


def draw_mask_on_img(mask, img, alpha = 0.6, color = (255, 0, 0)):
    """
    @brief Draws a mask on a 3-channel image using a given alpha and a give color
    """
    _, mask_inv = cv.threshold(mask, 128, 255, cv.THRESH_BINARY_INV)
    background = cv.bitwise_and(img, img, mask=mask_inv)

    # Blend image and mask
    film = np.zeros(img.shape, dtype=np.uint8)
    film[:] = color
    beta = 1.0 - alpha
    blended = cv.addWeighted(film, alpha, img, beta, gamma=0)
    foreground = cv.bitwise_and(blended, blended, mask=mask)
    img_with_mask = cv.add(foreground, background)
    return img_with_mask


def create_masks_from_meshes(meshes:list[np.ndarray], img:np.ndarray, rotation:np.ndarray, center:np.ndarray, camera_matrix:np.ndarray, dist_coeffs:np.ndarray) -> list[np.ndarray]:
    """
    @brief Projects N meshes onto an image creating a list of N masks
    """
    
    img_height = img.shape[0]
    img_width = img.shape[1]

    masks = []
    # project all vertices
    for mesh in meshes:
        tvec = -np.matmul(rotation, center)
        rvec, _ = cv.Rodrigues(rotation)
        # If experienced some problems with distorsion coefficient and points very much out of the image,
        # then keep a zero distorsion with:
        # zero_distorsion = np.array([0, 0, 0, 0], dtype = np.float32)
        # This problem should be solved if the meshes are cut with a camera fuse

        pixels_projected, _ = cv.projectPoints(mesh.vertices.astype(np.float32), rvec.astype(np.float32), tvec.astype(np.float32), camera_matrix.astype(np.float32), dist_coeffs.astype(np.float32))
        pixels_projected = pixels_projected.squeeze().astype(np.int32)
        # build an ideal grayscale image
        u_min = min(0, np.min(pixels_projected[:, 0]))
        u_max = max(img_width - 1, np.max(pixels_projected[:, 0]))
        v_min = min(0, np.min(pixels_projected[:, 1]))
        v_max = max(img_height - 1, np.max(pixels_projected[:, 1]))
        # Big mask containing all in grayscale
        big_mask_width = u_max - u_min + 1
        big_mask_height = v_max - v_min + 1
        # print(f"Creating mask with {big_mask_width, big_mask_height} shape")
        big_mask = np.zeros((big_mask_height, big_mask_width), dtype=np.uint8)
        # Translation vector moving the origin: how the big mask origin sees the original origin
        t_big_mask = np.int32([-u_min, -v_min])
        # Translate to have all the projected pixels positive in (u, v)
        pixels_projected_t = pixels_projected + t_big_mask
        # Draw the triangles
        for t in range(mesh.faces.shape[0]):
            poly = pixels_projected_t[mesh.faces[t, :]]
            cv.drawContours(image=big_mask, contours=[poly], contourIdx=0, color=255, thickness=-1)

        mask = crop(t_big_mask[0], t_big_mask[1], img_width, img_height, big_mask)
        masks.append(mask)

    return masks