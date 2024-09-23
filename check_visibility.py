import numpy as np
import cv2 as cv
import trimesh
import os
import json
from mesh_utils import slice_mesh_with_fuse

IMG_FOLDER_PATH = 'image_selection_data/P_1'
IMG_FORMAT = '.JPG'
POSES_FOLDER = 'poses'
MESH_PATH = 'image_selection_data/cut_decimated_centered_textured_mesh.obj'
CAMERAS_PATH = "cameras.json"


def get_camera_info(camera_id: str, camera_info_path=CAMERAS_PATH):
    """
    Retrieve camera information from a JSON file.
    
    Args:
        camera_id (str): The ID of the camera.
        camera_info_path (str): Path to the JSON file containing camera information.
    
    Returns:
        dict: Camera information.
    """
    assert os.path.exists(camera_info_path), f"Camera file not found: {camera_info_path}"
    with open(camera_info_path) as json_file:
        cameras_info = json.load(json_file)

    for camera_info in cameras_info.values():
        if camera_info["id"] == camera_id:
            return camera_info
    
    raise ValueError(f"Camera with id {camera_id} not found in {camera_info_path}")


def load_image_info(image_name: str):
    """
    Load image and pose information for a given image.
    
    Args:
        image_name (str): Name of the image file (without extension).
    
    Returns:
        tuple: Pose information and camera information.
    """
    # image_path = os.path.join(IMG_FOLDER_PATH, image_name + IMG_FORMAT)
    # assert os.path.exists(image_path), "Image not found"

    pose_path = os.path.join(POSES_FOLDER, image_name + '.json')
    assert os.path.exists(pose_path), "Pose file not found"
    with open(pose_path) as json_file:
        pose_info_dict = json.load(json_file)

    camera_info_dict = get_camera_info(pose_info_dict["camera_id"])
    return pose_info_dict, camera_info_dict


def load_and_prepare_mesh(mesh_path: str, camera_matrix: np.ndarray,
                          rotation: np.ndarray, center: np.ndarray, img_height: int, img_width: int):
    """
    Load the mesh and apply a slice based on the camera's field of view.
    
    Args:
    - mesh_path: Path to the mesh file.
    - camera_matrix: Intrinsic camera matrix.
    - rotation: Rotation matrix of the camera.
    - center: Translation vector (camera position).
    - img_height: Height of the camera image.
    - img_width: Width of the camera image.
    
    Returns:
    - pier: The loaded mesh object.
    - pier_cutted: The mesh object after slicing.
    """
    pier = trimesh.load(mesh_path, force='mesh')
    pier_cutted = slice_mesh_with_fuse(rotation, center, camera_matrix/10, int(img_height*2), int(img_width*2), pier)
    return pier, pier_cutted


def prepare_camera_and_pose_data(camera_info: dict, pose_info: dict):
    # Unpacks camera_info and pose_info and converts data to float64
    camera_matrix = np.float64(camera_info["matrix"])
    distortion_coefficients = np.float64(camera_info["distortion_coefficients"])
    rotation = np.float64(pose_info["rotation"]).reshape(3, 3)
    center = np.float64(pose_info["center"]).reshape(3, 1)
    img_height = camera_info["height"]
    img_width = camera_info["width"]
    return camera_matrix, distortion_coefficients, rotation, center, img_height, img_width


def get_vertices_behind_camera(vertices: np.ndarray, rotation: np.ndarray, translation_cam: np.ndarray):
    vertices = np.array(vertices, dtype=np.float64)
    rotation = np.array(rotation, dtype=np.float64)
    translation_cam = np.array(translation_cam, dtype=np.float64).squeeze()

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation_cam
    homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    camera_coords = np.dot(homogeneous_vertices, transform.T)
    behind_camera = camera_coords[:, 2] < 0

    return behind_camera


def project_vertices(mesh, rotation: np.ndarray, center: np.ndarray,
                     camera_matrix: np.ndarray, distortion_coefficients: np.ndarray) -> np.ndarray:
    """
    mesh: The mesh object containing the vertices to be projected.
    Returns:
     - The projected vertices as a numpy array.
    """
    translation_cam = (-rotation @ center).reshape(3, 1)
    projected_vertices, _ = cv.projectPoints(mesh.vertices.view(np.ndarray).astype(np.float64),
                                             np.float64(rotation), np.float64(translation_cam),
                                             np.float64(camera_matrix),
                                             np.float64(distortion_coefficients))
    projected_vertices = projected_vertices.squeeze()
    return projected_vertices


def filter_vertices(vertices: np.ndarray, projected_vertices: np.ndarray, img_height: np.ndarray,
                    img_width: np.ndarray, rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    # Filter vertices that are not in the camera view
    behind_camera = get_vertices_behind_camera(vertices, rotation, (-rotation @ center).reshape(3, 1))
    in_front_of_camera = ~behind_camera

    within_x_bounds = (projected_vertices[:, 0] >= 0) & (projected_vertices[:, 0] < img_width)
    within_y_bounds = (projected_vertices[:, 1] >= 0) & (projected_vertices[:, 1] < img_height)
    within_image_bounds = within_x_bounds & within_y_bounds
    potentially_visible = in_front_of_camera & within_image_bounds

    potential_indices = np.where(potentially_visible)[0]
    return potential_indices


def check_visibility(mesh, vertices: np.ndarray, potential_indices: np.ndarray, center: np.ndarray,
                     threshold=0.01, verbose=True)-> np.ndarray:
    """
        Check the visibility of the vertices by casting rays and determining which vertices
        are occluded or visible from the camera's position.

        Args:
        - mesh: The 3D mesh object containing vertices and faces.
        - vertices: 3D vertices of the mesh as numpy array.
        - potential_indices: Indices of vertices that are within the camera's view.
        - center: Camera position (translation vector).
        - verbose: Boolean flag to print additional debug information.

        Returns:
        - visible_indices: Indices of vertices that are visible from the camera's point of view.
        """
    vertices_projected = vertices[potential_indices]
    
    rays_directions = vertices_projected - center.squeeze()
    distances = np.linalg.norm(rays_directions, axis=1)
    rays_directions /= distances.reshape(-1, 1)
    rays_origins = np.tile(center.squeeze(), (len(rays_directions), 1))

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=rays_origins,
        ray_directions=rays_directions,
        multiple_hits=False
    )
    any_hit = mesh.ray.intersects_any(
        ray_origins=rays_origins,
        ray_directions=rays_directions
    )

    potential_indices_hits = np.where(any_hit)[0]
    vertices_projected_hits = vertices_projected[potential_indices_hits]
    # Here we check if the ray pointed to vertice hits mesh near that vertice
    ray_miss = np.abs(vertices_projected_hits - locations)
    ray_miss = ray_miss < threshold
    visible = np.logical_and(ray_miss[:, 0], np.logical_and(ray_miss[:, 1], ray_miss[:, 2]))
    visible_indices = potential_indices[potential_indices_hits[visible]]
    if verbose:
        print('Total vertices', len(vertices))
        print('Projected vertices', len(vertices_projected))
        print('Ray hits',len(potential_indices_hits))
        print('Visible', sum(visible))

    return visible_indices


def find_indices(original_points, slice_points):
    """
       Map indices from sliced mesh vertices to original mesh vertices.
       Returns:
       - indices: Indices of the sliced vertices in the original mesh.
       """
    indices = []
    for point in slice_points:
        index = np.where((original_points == point).all(axis=1))[0]
        if len(index) > 0:
            indices.append(index[0])
    
    return np.array(indices)


def get_visible_vertices(image_name: str, verbose=True):
    """
        Main function to find visible vertices from a mesh given an image.

        Args:
        - image_name: Name of the image file.
        - verbose: Whether to print additional information.

        Returns:
        - visible_indices_orig: Indices of visible vertices in the original mesh.
        - visible_indices: Indices of visible vertices in the sliced mesh.
        """
    pose_info, camera_info = load_image_info(image_name)
    camera_matrix, distortion_coefficients, rotation, center, img_height, img_width = prepare_camera_and_pose_data(camera_info, pose_info)
    
    pier, pier_cutted = load_and_prepare_mesh(MESH_PATH, camera_matrix, rotation, center, img_height, img_width)
    projected_vertices = project_vertices(pier_cutted, rotation, center, camera_matrix, distortion_coefficients)
    
    vertices = pier_cutted.vertices.view(np.ndarray).astype(np.float64)
    potential_indices = filter_vertices(vertices, projected_vertices, img_height, img_width, rotation, center)
    visible_indices = check_visibility(pier_cutted, vertices, potential_indices, center, verbose=verbose)

    # to create mapping from cutted mesh to original
    visible_vertices = vertices[visible_indices]
    vertices_original = pier.vertices.view(np.ndarray).astype(np.float64)
    if verbose:
        print('Total vertices original', len(vertices_original))
    visible_indices_orig = find_indices(vertices_original, visible_vertices)

    return visible_indices_orig, visible_indices

if __name__ == "__main__":
    image_name = 'DJI_20240417190632_0129_Z'
    visible_indices = get_visible_vertices(image_name)
    print(f"Number of visible vertices: {len(visible_indices)}")
    print(f"Visible vertex indices: {visible_indices}")