import numpy as np
from pathlib import Path
import torch
import os

def convert_matrix_world_to_RT(matrix_world):
    # bcam stands for blender camera
    R_bcam2cv = np.array([[1, 0,  0],
                          [0, -1, 0],
                          [0, 0, -1]])

    location = matrix_world[:3, 3]
    R_world2bcam = matrix_world[:3, :3].T

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision
    # camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    # RT = np.zeros((3, 4))
    RT = np.eye(4)
    RT[:3, :3] = R_world2cv
    RT[:3,  3] = T_world2cv

    return RT
def process2cameras(save_path):
    cameras = {}
    instance_dir = Path("/mnt/data/cxy_colmap/volsdf/data/blender")
    split = 'test'
    split_list = [x for x in instance_dir.iterdir() if x.stem.startswith(split)]
    for item_path in split_list:
        idx = int(item_path.name.split('_')[-1])
        import json
        item_meta_path = item_path / 'metadata.json'
        with open(item_meta_path, 'r') as f:
            meta = json.load(f)
        img_wh = (int(meta['imw']), int(meta['imh']))
        focal = 0.5 * int(meta['imw']) / np.tan(0.5 * meta['cam_angle_x'])  # fov -> focal length
        focal *= img_wh[0] / meta['imw']
        intrinsics = torch.tensor([[focal, 0, img_wh[0] / 2, 0], [0, focal, img_wh[1] / 2, 0], [0, 0, 1, 0]]).float()  # [3, 4]
        cam_trans = np.array(list(map(float, meta["cam_transform_mat"].split(',')))).reshape(4, 4)
        cam_RT = convert_matrix_world_to_RT(cam_trans)
        P = np.eye(4)
        P[:3, :] = intrinsics.numpy() @ cam_RT
        cameras['world_mat_%d' % idx] = P.copy()
        cameras['scale_mat_%d' % idx] = np.eye(4)
    save_path = os.path.join(save_path, "cameras_"+split)
    np.savez(save_path, **cameras)
        
        
if __name__ == '__main__':
    save_path = "/mnt/data/cxy_colmap/volsdf/data/cameras"
    process2cameras(save_path)
    
    