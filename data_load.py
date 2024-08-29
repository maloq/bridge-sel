import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import trimesh
import json

def correct_meshroom_extrinsics(camera_rotation_meshroom, camera_center_meshroom):
    rotation_180_around_x_correction = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float32)
    center = np.matmul(rotation_180_around_x_correction, camera_center_meshroom)
    rotation = np.matmul(camera_rotation_meshroom, rotation_180_around_x_correction)
    return rotation, center



def get_image_data_from_json(image_name, json_file_path):
    """ Loads image extrinsics using image name and given json file with sertain structure"""
    #load json as dict 
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    poses = data['poses']
    views = data['views']
    viewId = None
    poseId = None
    frameId = None
    intrinsicId = None
    
    for v in views:
        if v['path'].split('/')[-1] == image_name:
            viewId = v['viewId']
            poseId = v['poseId']
            frameId = v['frameId']
            intrinsicId = v['intrinsicId']
            # print(v['path'])
    if viewId is not None:
        for p in poses:
            if p['poseId'] == poseId:
                rotation = p['pose']['transform']['rotation']
                center = p['pose']['transform']['center']
    else:
        raise Exception(f'Image {image_name} is not found in {json_file_path}')
    
    return viewId, poseId, frameId, intrinsicId, rotation, center


def get_camera_intrinsics_from_json(image_name, json_file_path):
    """ Loads camera intrinsics using image name and given json file with sertain structure"""
    _, _, _, intrinsicId, _, _ = get_image_data_from_json(image_name, json_file_path)
    with open(json_file_path) as json_file:
        data = json.load(json_file)
    intrinsics = data['intrinsics']
    for i in intrinsics:
        if i['intrinsicId'] == intrinsicId:
            width = float(i['width'])
            height = float(i['height'])

            sensor_width = float(i['sensorWidth'])
            sensor_height = float(i['sensorHeight'])
            focal_length = float(i['focalLength'])

            offset_x = float(i['principalPoint'][0])
            offset_y = float(i['principalPoint'][1])
            fx = width / sensor_width * focal_length

            fy = height / sensor_height * focal_length
            cx = offset_x + width * 0.5
            cy = offset_y + height * 0.5
            k1 = float(i['distortionParams'][0])
            k2 = float(i['distortionParams'][1])
            k3 = float(i['distortionParams'][2])
            dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype = np.float32)

            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]
            ], dtype = np.float32)
            return camera_matrix, dist_coeffs