import torch
import torch.nn as nn
import time

import utils.general as utils
from utils import rend_util
from model.network import ImplicitNetwork, RenderingNetwork
from model.density import LaplaceDensity, AbsDensity
# from model.ray_sampler import ErrorBoundSampler
from model.my_ray_sampler import ErrorBoundSampler


"""
For modeling more complex backgrounds, we follow the inverted sphere parametrization from NeRF++ 
https://github.com/Kai-46/nerfplusplus 
"""


class VolSDFNetworkBG(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)

        # Foreground object's networks
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))

        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, inverse_sphere_bg=True, **conf.get_config('ray_sampler'))

        # Background's networks
        bg_feature_vector_size = conf.get_int('bg_network.feature_vector_size')
        self.bg_implicit_network = ImplicitNetwork(bg_feature_vector_size, 0.0, **conf.get_config('bg_network.implicit_network'))
        self.bg_rendering_network = RenderingNetwork(bg_feature_vector_size, **conf.get_config('bg_network.rendering_network'))
        self.bg_density = AbsDensity(**conf.get_config('bg_network.density', default={}))

    def forward(self, input):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape
        # import ipdb; ipdb.set_trace()
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_ret = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        
        points_1 = rend_util.get_points(z_ret["1"]["cam_loc"], z_ret["1"]["z_vals"], z_ret["1"]["ray_dirs"])
        points_2 = rend_util.get_points(z_ret["2"]["cam_loc"], z_ret["2"]["z_vals"], z_ret["2"]["ray_dirs"])
        points_3 = rend_util.get_points(z_ret["3"]["cam_loc"], z_ret["3"]["z_vals"], z_ret["3"]["ray_dirs"])
        
        mask = z_ret["mask"]
        N_rays = mask.shape[0]
        
        points_long = torch.cat([points_1[mask], points_2[mask], points_3[mask]], 1)
        points_long_flat = points_long.reshape(-1, 3)
        
        N_samples = z_ret["1"]["z_vals"].shape[1]
        
        dirs_long = torch.cat([z_ret["1"]["ray_dirs"][mask].unsqueeze(1).repeat(1, N_samples,1), 
                               z_ret["2"]["ray_dirs"][mask].unsqueeze(1).repeat(1, N_samples,1), 
                               z_ret["3"]["ray_dirs"].unsqueeze(1).repeat(1, N_samples,1)], 1)
        dirs_long_flat = dirs_long.reshape(-1, 3)
        
        z_max_1 = z_ret["1"]["z_vals"][:, -1]
        z_max_2 = z_ret["2"]["z_vals"][:, -1]
        z_max_3 = z_ret["3"]["z_vals"][:, -1]
        
        z_vals_long = torch.cat([z_ret["1"]["z_vals"][mask], 
                               z_ret["2"]["z_vals"][mask] + z_max_1[mask], 
                               z_ret["3"]["z_vals"] + z_max_1[mask] + z_max_2[mask]], 1)
        
        # long
        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_long_flat)
        rgb_flat = self.rendering_network(points_long_flat, gradients, dirs_long_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)
        
        z_max = z_max_1[mask] + z_max_2[mask] + z_max_3
        weights, bg_transmittance = self.volume_rendering(z_vals_long, z_max, sdf)
        fg_long_rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        
        # short
        mask = ~mask
        points_short = torch.cat([points_1[mask], points_2[mask]], 1)
        points_short_flat = points_long.reshape(-1, 3)
        dirs_long = torch.cat([z_ret["1"]["ray_dirs"][mask].unsqueeze(1).repeat(1, N_samples,1), 
                               z_ret["2"]["ray_dirs"][mask].unsqueeze(1).repeat(1, N_samples,1)], 1)
        dirs_long_flat = dirs_long.reshape(-1, 3)
        z_vals_long = torch.cat([z_ret["1"]["z_vals"][mask], 
                               z_ret["2"]["z_vals"][mask] + z_max_1[mask]], 1)
        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_short)
        rgb_flat = self.rendering_network(points_short_flat, gradients, dirs_long_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)
        
        z_max = z_max_1[mask] + z_max_2[mask]
        weights, bg_transmittance = self.volume_rendering(z_vals_long, z_max, sdf)
        fg_short_rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        
        fg_rgb_values = torch.zeros([N_rays, 3])
        fg_rgb_values[mask] = fg_short_rgb_values
        fg_rgb_values[~mask] = fg_long_rgb_values
        
        # z_vals, z_vals_bg = z_vals
        # z_max = z_vals[:,-1]
        # z_vals = z_vals[:,:-1]
        # N_samples = z_vals.shape[1]

        # points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        # points_flat = points.reshape(-1, 3)

        # dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        # dirs_flat = dirs.reshape(-1, 3)

        # sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat)

        # rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors)
        # rgb = rgb_flat.reshape(-1, N_samples, 3)

        # weights, bg_transmittance = self.volume_rendering(z_vals, z_max, sdf)

        # fg_rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        z_vals_bg = z_ret["bg"]["z_vals"]
        # Background rendering
        N_bg_samples = z_vals_bg.shape[1]
        z_vals_bg = torch.flip(z_vals_bg, dims=[-1, ])  # 1--->0

        bg_dirs = ray_dirs.unsqueeze(1).repeat(1,N_bg_samples,1)
        bg_locs = cam_loc.unsqueeze(1).repeat(1,N_bg_samples,1)

        bg_points = self.depth2pts_outside(bg_locs, bg_dirs, z_vals_bg)  # [..., N_samples, 4]
        bg_points_flat = bg_points.reshape(-1, 4)
        bg_dirs_flat = bg_dirs.reshape(-1, 3)

        output = self.bg_implicit_network(bg_points_flat)
        bg_sdf = output[:,:1]
        bg_feature_vectors = output[:, 1:]
        bg_rgb_flat = self.bg_rendering_network(None, None, bg_dirs_flat, bg_feature_vectors)
        bg_rgb = bg_rgb_flat.reshape(-1, N_bg_samples, 3)

        bg_weights = self.bg_volume_rendering(z_vals_bg, bg_sdf)

        bg_rgb_values = torch.sum(bg_weights.unsqueeze(-1) * bg_rgb, 1)


        # Composite foreground and background
        bg_rgb_values = bg_transmittance.unsqueeze(-1) * bg_rgb_values
        rgb_values = fg_rgb_values + bg_rgb_values

        output = {
            'rgb_values': rgb_values,
        }

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            # eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eik_near_points_1 = rend_util.get_points(z_ret["1"]["cam_loc"], z_ret["1"]["z_samples_eik"], z_ret["1"]["ray_dirs"])
            eik_near_points_2 = rend_util.get_points(z_ret["2"]["cam_loc"], z_ret["2"]["z_samples_eik"], z_ret["2"]["ray_dirs"])
            eik_near_points_3 = rend_util.get_points(z_ret["3"]["cam_loc"], z_ret["3"]["z_samples_eik"], z_ret["3"]["ray_dirs"])
            
            eik_near_points = torch.cat([eik_near_points_1, eik_near_points_2, eik_near_points_3], 0)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)

            grad_theta = self.implicit_network.gradient(eikonal_points)
            output['grad_theta'] = grad_theta

        if not self.training:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

            output['normal_map'] = normal_map

        return output

    def volume_rendering(self, z_vals, z_max, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1]) # (batch_size * num_pixels) x N_samples

        # included also the dist from the sphere intersection
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, z_max.unsqueeze(-1) - z_vals[:, -1:]], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy], dim=-1)  # add 0 for transperancy 1 at t_0
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        fg_transmittance = transmittance[:, :-1]
        weights = alpha * fg_transmittance  # probability of the ray hits something here
        bg_transmittance = transmittance[:, -1]  # factor to be multiplied with the bg volume rendering

        return weights, bg_transmittance

    def bg_volume_rendering(self, z_vals_bg, bg_sdf):
        bg_density_flat = self.bg_density(bg_sdf)
        bg_density = bg_density_flat.reshape(-1, z_vals_bg.shape[1]) # (batch_size * num_pixels) x N_samples

        bg_dists = z_vals_bg[:, :-1] - z_vals_bg[:, 1:]
        bg_dists = torch.cat([bg_dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(bg_dists.shape[0], 1)], -1)

        # LOG SPACE
        bg_free_energy = bg_dists * bg_density
        bg_shifted_free_energy = torch.cat([torch.zeros(bg_dists.shape[0], 1).cuda(), bg_free_energy[:, :-1]], dim=-1)  # shift one step
        bg_alpha = 1 - torch.exp(-bg_free_energy)  # probability of it is not empty here
        bg_transmittance = torch.exp(-torch.cumsum(bg_shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        bg_weights = bg_alpha * bg_transmittance # probability of the ray hits something here

        return bg_weights

    def depth2pts_outside(self, ray_o, ray_d, depth):

        '''
        ray_o, ray_d: [..., 3]
        depth: [...]; inverse of distance to sphere origin
        '''

        o_dot_d = torch.sum(ray_d * ray_o, dim=-1)
        under_sqrt = o_dot_d ** 2 - ((ray_o ** 2).sum(-1) - self.scene_bounding_sphere ** 2)
        d_sphere = torch.sqrt(under_sqrt) - o_dot_d
        p_sphere = ray_o + d_sphere.unsqueeze(-1) * ray_d
        p_mid = ray_o - o_dot_d.unsqueeze(-1) * ray_d
        p_mid_norm = torch.norm(p_mid, dim=-1)

        rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
        rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
        phi = torch.asin(p_mid_norm / self.scene_bounding_sphere)
        theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
        rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

        # now rotate p_sphere
        # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                       torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                       rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
        p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

        return pts