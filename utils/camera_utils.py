import os
import json
import numpy as np
import torch
import math
import torch.nn as nn

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
            Rt = np.zeros((4, 4))
            Rt[:3, :3] = R.transpose()
            Rt[:3, 3] = t
            Rt[3, 3] = 1.0

            C2W = np.linalg.inv(Rt)
            cam_center = C2W[:3, 3]
            cam_center = (cam_center + translate) * scale
            C2W[:3, 3] = cam_center
            Rt = np.linalg.inv(C2W)
            return np.float32(Rt)

        def getProjectionMatrix(znear, zfar, fovX, fovY):
            tanHalfFovY = math.tan((fovY / 2))
            tanHalfFovX = math.tan((fovX / 2))

            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right

            P = torch.zeros(4, 4)

            z_sign = 1.0

            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left)
            P[1, 2] = (top + bottom) / (top - bottom)
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P


        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def generate_camera_rotation_matrix(camera_to_object, object_vertical_downward):
    camera_to_object = camera_to_object / np.linalg.norm(
        camera_to_object
    )  # last column
    # the second column of rotation matrix is pointing toward the downward vertical direction
    camera_y = (
        object_vertical_downward
        - np.dot(object_vertical_downward, camera_to_object) * camera_to_object
    )
    camera_y = camera_y / np.linalg.norm(camera_y)  # second column
    first_column = np.cross(camera_y, camera_to_object)
    R = np.column_stack((first_column, camera_y, camera_to_object))
    return R


# supply vertical vector in world space
def generate_local_coord(vertical_vector):
    vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
    horizontal_1 = np.array([1, 1, 1])
    if np.abs(np.dot(horizontal_1, vertical_vector)) < 0.01:
        horizontal_1 = np.array([0.72, 0.37, -0.67])
    # gram schimit
    horizontal_1 = (
        horizontal_1 - np.dot(horizontal_1, vertical_vector) * vertical_vector
    )
    horizontal_1 = horizontal_1 / np.linalg.norm(horizontal_1)
    horizontal_2 = np.cross(horizontal_1, vertical_vector)

    return vertical_vector, horizontal_1, horizontal_2


# scalar (in degrees), scalar (in degrees), scalar, vec3, mat33 = [horizontal_1; horizontal_2; vertical];  -> vec3
def get_point_on_sphere(azimuth, elevation, radius, center, observant_coordinates):
    canonical_coordinates = (
        np.array(
            [
                np.cos(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
                np.sin(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
                np.sin(elevation / 180.0 * np.pi),
            ]
        )
        * radius
    )

    return center + observant_coordinates @ canonical_coordinates


def get_camera_position_and_rotation(
    azimuth, elevation, radius, view_center, observant_coordinates
):
    # get camera position
    position = get_point_on_sphere(
        azimuth, elevation, radius, view_center, observant_coordinates
    )
    # get rotation matrix
    R = generate_camera_rotation_matrix(
        view_center - position, -observant_coordinates[:, 2]
    )
    return position, R


def get_current_radius_azimuth_and_elevation(
    camera_position, view_center, observesant_coordinates
):
    center2camera = -view_center + camera_position
    radius = np.linalg.norm(center2camera)
    dot_product = np.dot(center2camera, observesant_coordinates[:, 2])
    cosine = dot_product / (
        np.linalg.norm(center2camera) * np.linalg.norm(observesant_coordinates[:, 2])
    )
    elevation = np.rad2deg(np.pi / 2.0 - np.arccos(cosine))
    proj_onto_hori = center2camera - dot_product * observesant_coordinates[:, 2]
    dot_product2 = np.dot(proj_onto_hori, observesant_coordinates[:, 0])
    cosine2 = dot_product2 / (
        np.linalg.norm(proj_onto_hori) * np.linalg.norm(observesant_coordinates[:, 0])
    )

    if np.dot(proj_onto_hori, observesant_coordinates[:, 1]) > 0:
        azimuth = np.rad2deg(np.arccos(cosine2))
    else:
        azimuth = -np.rad2deg(np.arccos(cosine2))
    return radius, azimuth, elevation

def get_camera_view(
    model_path='',
    default_camera_index=0,
    center_view_world_space=None,
    observant_coordinates=None,
    show_hint=False,
    init_azimuthm=None,
    init_elevation=None,
    init_radius=None,
    move_camera=False,
    current_frame=0,
    delta_a=0,
    delta_e=0,
    delta_r=0,
    resolution=1280,
    fov_scale=1.5
):
    """Load one of the default cameras for the scene."""
    cam_path = os.path.join(model_path, "cameras.json")
    if os.path.exists(cam_path):
        with open(cam_path) as f:
            data = json.load(f)

    if show_hint:
        if default_camera_index < 0:
            default_camera_index = 0
        r, a, e = get_current_radius_azimuth_and_elevation(
            data[default_camera_index]["position"],
            center_view_world_space,
            observant_coordinates,
        )
        print("Default camera ", default_camera_index, " has")
        print("azimuth:    ", a)
        print("elevation:  ", e)
        print("radius:     ", r)
        print("Now exit program and set your own input!")
        exit()

    if default_camera_index > -1:
        raw_camera = data[default_camera_index]

    elif default_camera_index == -1:
        # raw_camera = data[0]  # get data to be modified
        raw_camera = {}

        assert init_azimuthm is not None
        assert init_elevation is not None
        assert init_radius is not None

        if move_camera:
            assert delta_a is not None
            assert delta_e is not None
            assert delta_r is not None
            position, R = get_camera_position_and_rotation(
                init_azimuthm + current_frame * delta_a,
                init_elevation + current_frame * delta_e,
                init_radius + current_frame * delta_r,
                center_view_world_space,
                observant_coordinates,
            )
        else:
            position, R = get_camera_position_and_rotation(
                init_azimuthm,
                init_elevation,
                init_radius,
                center_view_world_space,
                observant_coordinates,
            )
        raw_camera["rotation"] = R.tolist()
        raw_camera["position"] = position.tolist()
    else:
        raw_camera = data[current_frame] 
    
    tmp = np.zeros((4, 4))
    tmp[:3, :3] = raw_camera["rotation"]
    tmp[:3, 3] = raw_camera["position"]
    tmp[3, 3] = 1
    C2W = np.linalg.inv(tmp)
    R = C2W[:3, :3].transpose()
    T = C2W[:3, 3]

    raw_camera['width'] = resolution
    raw_camera['height'] = resolution
    raw_camera['fx'] = resolution * fov_scale
    raw_camera['fy'] = resolution * fov_scale
    fovx = focal2fov(raw_camera['fx'], raw_camera['width'])
    fovy = focal2fov(raw_camera['fy'], raw_camera['height'])

    return Camera(
        colmap_id=0,
        R=R,
        T=T,
        FoVx=fovx,
        FoVy=fovy,
        image=torch.zeros((3, raw_camera['height'], raw_camera['width'])),  # fake
        gt_alpha_mask=None,
        image_name="fake",
        uid=0,
    ), raw_camera

def _extract_intrinsic_matrix(camera):

    width = camera.image_width
    height = camera.image_height
    
    fx = width / (2 * math.tan(camera.FoVx * 0.5))
    fy = height / (2 * math.tan(camera.FoVy * 0.5))
    
    cx = width / 2
    cy = height / 2
    
    intrinsic_matrix = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    return intrinsic_matrix


def _extract_extrinsic_matrix(camera):

    R_w2c = torch.tensor(camera.R.T, dtype=torch.float32)  # (3, 3)
    T_w2c = torch.tensor(camera.T, dtype=torch.float32)  # (3,)

    extrinsic_matrix = torch.zeros(4, 4, dtype=torch.float32)
    extrinsic_matrix[:3, :3] = R_w2c
    extrinsic_matrix[:3, 3] = T_w2c
    extrinsic_matrix[3, 3] = 1.0

    return extrinsic_matrix


def _extract_single_camera_matrices(camera):
    """
    Args:
        camera: GSCamera
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - intrinsic_matrix: (3, 3)
            - extrinsic_matrix: (4, 4)
    """
    intrinsic_matrix = _extract_intrinsic_matrix(camera)
    
    extrinsic_matrix = _extract_extrinsic_matrix(camera)
    
    return intrinsic_matrix, extrinsic_matrix

def extract_camera_matrices(cameras):
    """
    
    Args:
        cameras: GSCamera or GSCamera list
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - intrinsic_matrices: (V, 3, 3) or (1, 3, 3)
            - extrinsic_matrices: (V, 4, 4) or (1, 4, 4)
    """

    if isinstance(cameras, Camera):
        intrinsic, extrinsic = _extract_single_camera_matrices(cameras)
        return intrinsic.unsqueeze(0), extrinsic.unsqueeze(0)
    
    elif isinstance(cameras, list):
        if len(cameras) == 0:
            raise ValueError("Camera list cannot be empty")
        
        intrinsic_list = []
        extrinsic_list = []
        
        for camera in cameras:
            if not isinstance(camera, Camera):
                raise ValueError("Must be CSCamera objects")
            
            intrinsic, extrinsic = _extract_single_camera_matrices(camera)
            intrinsic_list.append(intrinsic)
            extrinsic_list.append(extrinsic)
        
        intrinsic_matrices = torch.stack(intrinsic_list, dim=0)
        extrinsic_matrices = torch.stack(extrinsic_list, dim=0)
        
        return intrinsic_matrices, extrinsic_matrices
    
    else:
        raise ValueError("Must be CSCamera objects or lists")
