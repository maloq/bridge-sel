import uuid
import json
import xml.etree.ElementTree as ET
import numpy as np


# Parse the cameras.xml file and returns the camera


CAMERAS_XML_PATH = "/home/teshbek/Work/Bridges_Project/Bridge-Selection/image_selection_data/cameras.xml"
OUTPUT_PATH = "cameras.json"
OUTPUT_MAPPING_PATH = "mapping.json"


def create_metashape_cameras(sfm_file_path: str):
    """Creates all the camera files in the output_dir"""

    sfm_file_stream = None
    with open(sfm_file_path) as f:
        sfm_file_stream = f.read()

    root = ET.fromstring(sfm_file_stream)

    camera_original_id_to_uuid = {}
    cameras_data = {}
    for sensor in root.iter("sensor"):
        if type(sensor) is not ET.Element:
            raise Exception("Invalid sensor type")
        sfm_original_id = sensor.get("id", None)
        if sfm_original_id is None:
            raise Exception("Missing sensor id")
        if type(sfm_original_id) is not str:
            raise Exception("Invalid id type")

        calibration = sensor.find("calibration")
        if calibration is None:
            print(f"Skipping sensor having id {sfm_original_id} because the calibration is not present")
            continue

        resolution_v0 = sensor.find("resolution")
        if resolution_v0 is None:
            raise Exception("Missing resolution_v0")
        resolution_width_v0_str = resolution_v0.get("width", None)
        if resolution_width_v0_str is None:
            raise Exception("Missing resolution v0 width string")
        if type(resolution_width_v0_str) is not str:
            raise Exception("Invalid resolution v0 width type")
        if not resolution_width_v0_str.isnumeric():
            raise Exception("Invalid resolution v0 width string")
        resolution_width_v0 = int(resolution_width_v0_str)

        resolution_height_v0_str = resolution_v0.get("height", None)
        if resolution_height_v0_str is None:
            raise Exception("Missing resolution v0 height string")
        if type(resolution_height_v0_str) is not str:
            raise Exception("Invalid resolution v0 height type")
        if not resolution_height_v0_str.isnumeric():
            raise Exception("Invalid resolution v0 height string")
        resolution_height_v0 = int(resolution_height_v0_str)

        resolution = calibration.find("resolution")
        if resolution is None:
            raise Exception("Missing resolution")

        resolution_width_str = resolution.get("width", None)
        if resolution_width_str is None:
            raise Exception("Missing resolution width string")
        if type(resolution_width_str) is not str:
            raise Exception("Invalid resolution width type")
        if not resolution_width_str.isnumeric():
            raise Exception("Invalid resolution width string")
        resolution_width = int(resolution_width_str)
        if resolution_width_v0 != resolution_width:
            raise Exception("Inconsistent resolution widths")

        resolution_height_str = resolution.get("height", None)
        if resolution_height_str is None:
            raise Exception("Missing resolution height string")
        if type(resolution_height_str) is not str:
            raise Exception("Invalid resolution height type")
        if not resolution_height_str.isnumeric():
            raise Exception("Invalid resolution height string")
        resolution_height = int(resolution_height_str)
        if resolution_height_v0 != resolution_height:
            raise Exception("Inconsistent resolution heights")

        focal_length = calibration.find("f")
        if focal_length is None:
            raise Exception("Missing focal length")
        focal_length = float(focal_length.text)

        # NOTE metashape cx is not consistent with OpenCV documentation (see initUndistortRectifyMap() documentation)
        #      According to OpenCV documentation it should be the offset_x
        cx = calibration.find("cx")
        if cx is None:
            cx = 0.0
        else:
            cx = float(cx.text)

        # NOTE The same inconsistency found for cx (see above)
        cy = calibration.find("cy")
        if cy is None:
            cy = 0.0
        else:
            cy = float(cy.text)

        # skew params b1 and b2 not considered for now

        k1 = calibration.find("k1")
        if k1 is None:
            k1 = 0.0
        else:
            k1 = float(k1.text)

        k2 = calibration.find("k2")
        if k2 is None:
            k2 = 0.0
        else:
            k2 = float(k2.text)

        k3 = calibration.find("k3")
        if k3 is None:
            k3 = 0.0
        else:
            k3 = float(k3.text)

        p1 = calibration.find("p1")
        if p1 is None:
            p1 = 0.0
        else:
            p1 = float(p1.text)

        p2 = calibration.find("p2")
        if p2 is None:
            p2 = 0.0
        else:
            p2 = float(p2.text)

        camera_info = {}
        # Assumption: f for metashape sensors is already in pixels, not in mm.
        #             So, there is no need for sensor_width nor sensor_height.
        # camera_info['sensor_width'] = None
        # camera_info['sensor_height'] = None
        camera_info["id"] = str(uuid.uuid4())
        camera_info["offset_x"] = cx
        camera_info["offset_y"] = cy
        camera_info["focal_length"] = focal_length
        camera_info["k1"] = k1
        camera_info["k2"] = k2
        camera_info["k3"] = k3
        camera_info["p1"] = p1
        camera_info["p2"] = p2
        camera_info["width"] = resolution_width
        camera_info["height"] = resolution_height

        fx = focal_length
        fy = focal_length
        new_cx = cx + resolution_width * 0.5
        new_cy = cy + resolution_height * 0.5
        matrix = np.array([[fx, 0, new_cx], [0, fy, new_cy], [0, 0, 1]], dtype=np.float64)
        camera_info["matrix"] = matrix.tolist()

        # Keep the order needed by OpenCV
        # (k1, k2, p1, p2[, k3[, ...),
        # see https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
        distortion_coefficients = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        camera_info["distortion_coefficients"] = distortion_coefficients.tolist()

        if sfm_original_id in camera_original_id_to_uuid:
            raise Exception(f"Duplicate id found: {sfm_original_id}")
        camera_original_id_to_uuid[sfm_original_id] = camera_info["id"]
        cameras_data[sfm_original_id] = camera_info

    print(f"Created {len(camera_original_id_to_uuid)} cameras")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(cameras_data, f, indent=4)

    image_name_to_camera_uuid = {}
    for camera in root.iter("camera"):
        if type(camera) is not ET.Element:
            return (None, "Invalid camera type")

        pose_id = camera.get("id", None)
        if pose_id is None:
            return (None, "Missing pose id")
        if type(pose_id) is not str:
            return (None, "Invalid pose id type")

        image_name = camera.get("label", None)
        if image_name is None:
            raise Exception("Missing label")
        if type(image_name) is not str:
            raise Exception("Invalid label type")

        sfm_original_camera_id = camera.get("sensor_id", None)
        if sfm_original_camera_id is None:
            raise Exception("Missing sensor id")
        if type(sfm_original_camera_id) is not str:
            raise Exception("Invalid sensor id type")
        if image_name in image_name_to_camera_uuid:
            raise Exception(f"Image {image_name} already in the dict, duplicate found")

        camera_uuid = camera_original_id_to_uuid.get(sfm_original_camera_id, None)
        if camera_uuid is None:
            raise Exception(f"Cannot find the corresponding uuid of camera {sfm_original_camera_id}")
        image_name_to_camera_uuid[image_name] = camera_uuid

    return image_name_to_camera_uuid


if __name__ == "__main__":
    images_to_cameras = create_metashape_cameras(CAMERAS_XML_PATH)

    print("Starting the creation of the output file")

    with open(OUTPUT_MAPPING_PATH, "w") as f:
        json.dump(images_to_cameras, f, indent=4)

    print("Done")

