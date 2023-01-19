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
        pose = cam_trans @ self.blender2opencv
        c2w = torch.FloatTensor(pose)  # [4, 4]

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
