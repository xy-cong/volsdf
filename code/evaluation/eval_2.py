import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import skimage

import imageio
import time

import utils.general as utils
import utils.plots as plt
from utils import rend_util

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    
    R_bcam2cv = torch.from_numpy(np.array([[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]])).float()
    # import ipdb; ipdb.set_trace()
    location = c2w[:3, 3]
    R_world2bcam = c2w[:3, :3].T

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision
    # camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    c2w[:3, :3] = R_world2cv
    c2w[:3,  3] = T_world2cv
    return c2w

def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']
    eval_rendering = kwargs['eval_rendering']
    render_video = kwargs['render_video']

    expname = conf.get_string('train.expname') + kwargs['expname']
    # import ipdb; ipdb.set_trace()
    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            # self.timestamp = sorted(timestamps)[-1]
            timestamp = None
            for t in sorted(timestamps):
                if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname, t, 'checkpoints',
                                               'ModelParameters', str(kwargs['checkpoint']) + ".pth")):
                    timestamp = t
            if timestamp is None:
                print('NO GOOD TIMSTAMP')
                exit()
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']
    
    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname)
    utils.mkdir_ifnotexists(evaldir)

    dataset_conf = conf.get_config('dataset')
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)

    conf_model = conf.get_config('model')
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf_model)
    if torch.cuda.is_available():
        model.cuda()
    # import ipdb; ipdb.set_trace()
 
    if eval_rendering:
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )
        total_pixels = eval_dataset.total_pixels
        img_res = eval_dataset.img_res
        split_n_pixels = conf.get_int('train.split_n_pixels', 10000)

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    print("old_checkpnts_dir", old_checkpnts_dir)
    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
    model.load_state_dict(saved_model_state["model_state_dict"])
    epoch = saved_model_state['epoch']
    print("epoch: ", epoch)

    ####################################################################################################################
    print("evaluating...")

    model.eval()
    images_dir = '{0}/rendering_{1}'.format(evaldir, epoch)
    utils.mkdir_ifnotexists(images_dir)
    rgbs = []

    if render_video:
        print('RENDER VIDEO')
            # Default is smoother render_poses path
        images = None
        video_dir = '{0}/rendering_{1}'.format(evaldir, epoch)
        utils.mkdir_ifnotexists(video_dir)

        # interval = 40
        # render_poses = torch.stack([pose_spherical(angle, -30.0, 3.0) for angle in np.linspace(-180,180,interval+1)[:-1]], 0)
        # print('render_videos poses shape', render_poses.shape)
        
        # from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图     
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # r_x = [c[:3, 3][0] for c in render_poses]
        # r_y = [c[:3, 3][1] for c in render_poses]
        # r_z = [c[:3, 3][2] for c in render_poses]
        # import ipdb; ipdb.set_trace()
        # ax.scatter(r_x, r_y, r_z)
        # plt.savefig("b.jpg")
        # exit()
        
        # rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
        # rgbs = []
        
        # t = time.time()
        # for i, c2w in enumerate(tqdm(render_poses)):
        #     print(i, time.time() - t)
        #     t = time.time()
            
        #     model_input = {}
        #     model_input["intrinsics"] = eval_dataset.intrinsics_all[0].cuda()
        #     model_input["intrinsics"][0][1] = 0.0
        #     model_input["intrinsics"] = model_input["intrinsics"].reshape(1, 4, 4)
        #     # model_input["intrinsics"].requires_grad=True
        #     uv = np.mgrid[0:eval_dataset.img_res[0], 0:eval_dataset.img_res[1]].astype(np.int32)
        #     uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        #     uv = uv.reshape(2, -1).transpose(1, 0)
        #     # import ipdb; ipdb.set_trace()
        #     model_input["uv"] = uv.reshape(1, total_pixels, 2).cuda()
        #     # model_input["uv"].requires_grad=True
        #     model_input['pose'] = c2w.reshape(1,4,4).cuda()
        #     # model_input["pose"].requires_grad=True
        #     split = utils.split_input(model_input, total_pixels, n_pixels=split_n_pixels)
        #     res = []
        #     for s in tqdm(split):
        #         torch.cuda.empty_cache()
        #         out = model(s)
        #         res.append({
        #             'rgb_values': out['rgb_values'].detach(),
        #         })

        #     batch_size = model_input['pose'].shape[0]
        #     model_outputs = utils.merge_output(res, total_pixels, batch_size)
        #     rgb_eval = model_outputs['rgb_values']
        #     rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)

        #     rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
        #     rgb_eval = rgb_eval.transpose(1, 2, 0)
        #     img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
        #     img.save('{0}/eval_{1}.png'.format(images_dir,'%03d' % i))
        #     rgbs.append(rgb_eval)
        
        rgbs = []
        image_path = sorted(utils.glob_imgs(video_dir))
        for path in image_path:
            img = imageio.imread(path)
            img = skimage.img_as_float32(img)
            rgbs.append(img)
        
        to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
        
        imageio.mimwrite(os.path.join(images_dir, 'video.mp4'), to8b(rgbs), fps=5, quality=8)
        print('Done rendering', images_dir)

        return
    
    
    if eval_rendering:
        # import ipdb; ipdb.set_trace()
        camera_path = "/mnt/data/cxy_colmap/volsdf/data/blender/camera_path"
        intrinsics_folder = os.path.join(camera_path, "intrinsics")
        pose_folder = os.path.join(camera_path, "pose")
        intrinsics_path = utils.glob_par(intrinsics_folder, ['*.txt'])
        pose_path = utils.glob_par(pose_folder, ['*.txt'])
        intrinsics = []
        pose = []
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        for idx in range(len(intrinsics_path)):
            uv = np.mgrid[0:img_res[0], 0:img_res[1]].astype(np.int32)
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
            uv = uv.reshape(2, -1).transpose(1, 0).reshape(1, -1, 2)
            # import ipdb; ipdb.set_trace()
            # for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
            #     model_input["intrinsics"] = model_input["intrinsics"].cuda()
            #     model_input["uv"] = model_input["uv"].cuda()
            #     model_input['pose'] = model_input['pose'].cuda()
            intrinsics_file = open(intrinsics_path[idx])
            pose_file = open(pose_path[idx])
            intrin = intrinsics_file.readline().split(" ")
            intrin = np.array([float(i) for i in intrin]).reshape(4,4)
            p = pose_file.readline().split(" ")
            p = np.array([float(i) for i in p]).reshape(4,4)
            
            RT = utils.convert_matrix_world_to_RT(p)
            model_input = {}
            model_input["intrinsics"] = torch.from_numpy(intrin.reshape(1,4,4)).float().cuda()
            model_input["uv"] = uv.cuda()
            model_input['pose'] = torch.from_numpy(p.reshape(1,4,4)).float().cuda()
            # split_n_pixels = 1
            split = utils.split_input(model_input, total_pixels, n_pixels=split_n_pixels)
            res = []
            for s in tqdm(split):
                torch.cuda.empty_cache()
                out = model(s)
                res.append({
                    'rgb_values': out['rgb_values'].detach(),
                })
            # batch_size = ground_truth['rgb'].shape[0]
            batch_size = 1
            model_outputs = utils.merge_output(res, total_pixels, batch_size)
            rgb_eval = model_outputs['rgb_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)

            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/eval_{1}.png'.format(images_dir,'%03d' % idx))

            rgbs.append(rgb_eval)
            
            # import ipdb; ipdb.set_trace()
            
        # import ipdb; ipdb.set_trace()
        # psnrs = []
        # for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
        #     model_input["intrinsics"] = model_input["intrinsics"].cuda()
        #     model_input["uv"] = model_input["uv"].cuda()
        #     model_input['pose'] = model_input['pose'].cuda()
        #     # import ipdb; ipdb.set_trace()
        #     split = utils.split_input(model_input, total_pixels, n_pixels=split_n_pixels)
        #     res = []
        #     for s in tqdm(split):
        #         torch.cuda.empty_cache()
        #         out = model(s)
        #         res.append({
        #             'rgb_values': out['rgb_values'].detach(),
        #         })
            
        #     batch_size = ground_truth['rgb'].shape[0]
        #     model_outputs = utils.merge_output(res, total_pixels, batch_size)
        #     rgb_eval = model_outputs['rgb_values']
        #     rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)

        #     rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
        #     rgb_eval = rgb_eval.transpose(1, 2, 0)
        #     img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
        #     img.save('{0}/eval_{1}.png'.format(images_dir,'%03d' % indices[0]))

        #     psnr = rend_util.get_psnr(model_outputs['rgb_values'],
        #                               ground_truth['rgb'].cuda().reshape(-1, 3)).item()
        #     psnrs.append(psnr)
        #     rgbs.append(rgb_eval)
        print('Done rendering', images_dir)
        to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
        
        imageio.mimwrite(os.path.join(images_dir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--evals_folder', type=str, default='evals', help='The evaluation folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--eval_rendering', default=False, action="store_true", help='If set, evaluate rendering quality.')
    parser.add_argument('--render_video', default=False, action="store_true", help='If set, render video.')
    
    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=4, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        """
        order 不同 limit?
        """
        gpu = deviceIDs[0]
        print("GPU: ", gpu)
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate(conf=opt.conf,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name=opt.evals_folder,
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             resolution=opt.resolution,
             eval_rendering=opt.eval_rendering,
             render_video=opt.render_video
             )

