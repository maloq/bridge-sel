import os
import json
import numpy as np


# "pose" is equal to one image -> the position of the camera when the photo was taken


MAPPING_PATH = "mapping.json"
OPHIKAPPA_POSES_PATH = "image_selection_data/ophikappa_poses_centered.txt"
OUTPUT_FOLDER_PATH = "poses"


def split_metashape_poses_into_jsons(
    images_to_cameras_path: str, ophikappa_file_path: str, output_dir: str
) -> tuple[dict, str]:
    """Creates all the camera files in the output_dir"""

    if images_to_cameras_path is None:
        raise Exception("Invalid sfm_file_path")
    if not os.path.exists(images_to_cameras_path):
        raise Exception("Invalid sfm_file_path: not existing")
    if not os.path.isfile(images_to_cameras_path):
        raise Exception("Invalid sfm_file_path: not a file")
    if not os.path.exists(output_dir):
        raise Exception("Invalid output_dir: not existing")
    if not os.path.isdir(output_dir):
        raise Exception("Invalid output_dir: not a directory")

    image_names_to_camera_uuid = None
    with open(images_to_cameras_path) as f:
        image_names_to_camera_uuid = json.load(f)
    if image_names_to_camera_uuid is None:
        raise Exception("Invalid image_names_to_camera_uuid dict")

    rotation_180_x = np.float64([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    counter = 0
    with open(ophikappa_file_path) as f:
        while True:
            line = f.readline()
            line = line.rstrip(" \t\r\n")
            if line is None:
                break
            if len(line) == 0:
                break
            if line[0] == "#":
                continue
            tokens = line.split("\t")
            if len(tokens) != 16:
                raise Exception("Invalid length of tokens")
            image_name = tokens[0]
            translation_ophikappa = np.fromstring(" ".join(tokens[1:4]), sep=" ", dtype=np.float64).ravel()
            # ophikappa do not follow the usual convention, a 180 degrees rotation is needed
            rotation_ophikappa = np.fromstring(" ".join(tokens[7:]), sep=" ", dtype=np.float64).reshape(3, 3)
            rotation_ophikappa = rotation_180_x @ rotation_ophikappa

            pose_info = {}
            pose_info["center_x"] = translation_ophikappa[0]
            pose_info["center_y"] = translation_ophikappa[1]
            pose_info["center_z"] = translation_ophikappa[2]
            pose_info["rotation_00"] = rotation_ophikappa[0, 0]
            pose_info["rotation_01"] = rotation_ophikappa[0, 1]
            pose_info["rotation_02"] = rotation_ophikappa[0, 2]
            pose_info["rotation_10"] = rotation_ophikappa[1, 0]
            pose_info["rotation_11"] = rotation_ophikappa[1, 1]
            pose_info["rotation_12"] = rotation_ophikappa[1, 2]
            pose_info["rotation_20"] = rotation_ophikappa[2, 0]
            pose_info["rotation_21"] = rotation_ophikappa[2, 1]
            pose_info["rotation_22"] = rotation_ophikappa[2, 2]
            pose_info["image_original_name"] = image_name

            # Assemble them
            pose_info["center"] = [float(pose_info["center_x"]), float(pose_info["center_y"]), float(pose_info["center_z"])]
            pose_info["rotation"] = [
                [float(pose_info["rotation_00"]), float(pose_info["rotation_01"]), float(pose_info["rotation_02"])],
                [float(pose_info["rotation_10"]), float(pose_info["rotation_11"]), float(pose_info["rotation_12"])],
                [float(pose_info["rotation_20"]), float(pose_info["rotation_21"]), float(pose_info["rotation_22"])],
            ]

            camera_id = image_names_to_camera_uuid.get(image_name, None)
            if camera_id is None:
                print(f"Cannot find {image_name} in the {images_to_cameras_path} file")
                continue
            pose_info["camera_id"] = camera_id

            pose_json_path = os.path.join(output_dir, image_name + ".json")
            if os.path.exists(pose_json_path):
                print(f"ERROR: Cannot add pose {image_name} because already present")
                continue

            with open(pose_json_path, "w") as f_pose_json_path:
                json.dump(pose_info, f_pose_json_path, indent=4)

            counter += 1

    return counter


if __name__ == "__main__":
    print("Starting pose splitting")
    poses_nr = split_metashape_poses_into_jsons(
        MAPPING_PATH, OPHIKAPPA_POSES_PATH, OUTPUT_FOLDER_PATH
    )
    print(f"Done. Splitted {poses_nr} poses")
