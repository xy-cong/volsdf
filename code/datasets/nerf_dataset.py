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

        # self.convert_normal()
        # ---------------------------------------------------------------------- #
        # import ipdb; ipdb.set_trace()   
        self.cameras_file = os.path.join(data_dir, "cameras_normalized_{}.npz".format(self.split))
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
    
    def __len__(self):
        return self.n_images
    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        # import ipdb; ipdb.set_trace()
        item_path = self.split_list[idx]
        idx = int(item_path.name.split('_')[-1])
        intrinsics = self.intrinsics_all[idx]
        pose = self.pose_all[idx]
        c2w = torch.FloatTensor(pose)

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


