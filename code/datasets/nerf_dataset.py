import os
import torch
import numpy as np 

import utils.general as utils
from utils import rend_util
from pathlib import Path
class SceneDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 split='test',
                 ):

        self.instance_dir = Path(data_dir)
        print(self.instance_dir)

        self.img_res = (800,800)
        self.total_pixels=800*800
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None

        self.split = split
        self.split_list = [x for x in self.instance_dir.iterdir() if x.stem.startswith(self.split)]
        self.n_images = len(self.split_list)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # self.convert_normal()
        # ---------------------------------------------------------------------- #
        self.cameras_file = data_dir + "/cameras_normalize.npz"
        camera_dict = np.load(self.cameras_file)
        
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        
        self.intrinsics_all = []
        self.pose_all = []
        center = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            # import ipdb; ipdb.set_trace()
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            R = pose[:3, :3]
            t = pose[:3, 3]
            r = - R @ t
            if np.linalg.norm(r)<1 or np.linalg.norm(r)>3:
                import ipdb; ipdb.set_trace()
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
            center.append(t)
        # ---------------------------------------------------------------------- #
        # from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图     
        # import matplotlib.pyplot as plt

        # # 绘制散点图
        # fig = plt.figure()
        # ax = Axes3D(fig)

        # u = np.linspace(0, 2 * np.pi, 100)# 用参数方程画图
        # v = np.linspace(0, np.pi, 100)
        # r = 1
        # r_x = r * np.outer(np.cos(u), np.sin(v))# outer()外积函数：返回cosu × sinv
        # r_y = r * np.outer(np.sin(u), np.sin(v))# 
        # r_z = r * np.outer(np.ones(np.size(u)), np.cos(v))# ones()：返回一组[1,1,.......]

        # ax.plot_surface(r_x, r_y, r_z)
        
        # x = [xx[0] for xx in center]
        # y = [yy[1] for yy in center]
        # z = [zz[2] for zz in center]
        # ax.scatter(x, y, z)
        # plt.savefig("a.jpg")
        
        
        # # x = x.extend(r_x)
        # # y = y.extend(r_y)
        # # z = z.extend(r_z)
        # vertices = []
        # import ipdb; ipdb.set_trace()
        # for i in range(len(x)):
        #     vertices.append([x[i], y[i], z[i]])
        # output_file = 'Camera.ply'
        # b = np.float32(vertices)
        # one = np.ones((len(vertices),3))
        # one = np.float32(one)*255
        # print("\n Creating the output file... \n")
        # create_output(b, one, output_file)
            
    def convert_normal(self):
        cameras = {}
        print("Split: ", self.split)
        output_cameras_filename = "camera_new_" + self.split
        for idx in range(self.n_images):
            item_path = self.split_list[idx]
            import json
            item_meta_path = item_path / 'metadata.json'
            with open(item_meta_path, 'r') as f:
                meta = json.load(f)
            img_wh = (int(meta['imw'] ), int(meta['imh']))
            self.total_pixels = img_wh[0]*img_wh[1]

            # Get ray directions for all pixels, same for all images (with same H, W, focal)
            focal = 0.5 * int(meta['imw']) / np.tan(0.5 * meta['cam_angle_x'])  # fov -> focal length
            focal *= img_wh[0] / meta['imw']
            intrinsics = torch.tensor([[focal, 0, img_wh[0] / 2, 0], [0, focal, img_wh[1] / 2, 0], [0, 0, 1, 0]]).float()  # [3, 4]
            # TODO should change if update metadata.json cam_trans
            cam_trans = np.array(list(map(float, meta["cam_transform_mat"].split(',')))).reshape(4, 4)
            # pose = cam_trans @ self.blender2opencv
            cam_RT = convert_matrix_world_to_RT(cam_trans) # (4,4)
            P = np.eye(4)
            # import ipdb; ipdb.set_trace()
            P[:3, :] = intrinsics.numpy() @ cam_RT
            c2w = torch.FloatTensor(cam_RT)  # [4, 4]
            
            # R = c2w[:3, :3]
            # t = c2w[:3, 3]
            # world_mat = R @ t
            cameras['world_mat_%d' % idx] = P.copy()
            cameras['scale_mat_%d' % idx] = np.eye(4)
        
        np.savez(output_cameras_filename, **cameras)
        import ipdb; ipdb.set_trace()
    
    def __len__(self):
        return self.n_images
    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        item_path = self.split_list[idx]
        import json
        item_meta_path = item_path / 'metadata.json'
        with open(item_meta_path, 'r') as f:
            meta = json.load(f)
        img_wh = (int(meta['imw'] ), int(meta['imh']))
        self.total_pixels = img_wh[0]*img_wh[1]

        # Get ray directions for all pixels, same for all images (with same H, W, focal)
        focal = 0.5 * int(meta['imw']) / np.tan(0.5 * meta['cam_angle_x'])  # fov -> focal length
        focal *= img_wh[0] / meta['imw']
        intrinsics = torch.tensor([[focal, 0, img_wh[0] / 2], [0, focal, img_wh[1] / 2], [0, 0, 1]]).float()  # [3, 3]
        # TODO should change if update metadata.json cam_trans
        cam_trans = np.array(list(map(float, meta["cam_transform_mat"].split(',')))).reshape(4, 4)
        # pose = cam_trans @ self.blender2opencv
        # c2w = torch.FloatTensor(pose)  # [4, 4]
        intrinsics = self.intrinsics_all[idx]
        pose = self.pose_all[idx]
        c2w = torch.FloatTensor(pose)
        
        """
        c2w: [R|t]
        求RT 然后算距离, -R的转置 * t
        看看相机是不是在单位球外
        相机要normalize一下
        """
        # import ipdb; ipdb.set_trace()

        img_path = item_path / f'rgba.png'
        rgb = rend_util.load_rgb(img_path)
        ## Blend A to RGB
        
        rgb = torch.from_numpy(rgb.reshape(4, -1).transpose(1, 0)).float()
        rgb = rgb[:, :3] * rgb[:, -1:] + (1 - rgb[:, -1:])  # [3,H,W]

        sample = {
            "uv": uv,
            "intrinsics": intrinsics,
            "pose": c2w
        }

        ground_truth = {
            "rgb": rgb
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = rgb[self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]


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

# Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)