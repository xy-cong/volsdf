import abc
import torch

from utils import rend_util

class RaySampler(metaclass=abc.ABCMeta):
    def __init__(self,near, far):
        self.near = near
        self.far = far

    @abc.abstractmethod
    def get_z_vals(self, ray_dirs, cam_loc, model):
        pass

class UniformSampler(RaySampler):
    def __init__(self, scene_bounding_sphere, near, N_samples, take_sphere_intersection=False, far=-1,
                 sdf_threshold=5.0e-5, sphere_tracing_iters=0.5, line_step_iters=1, line_search_step=10):
        super().__init__(near, 2.0 * scene_bounding_sphere if far == -1 else far)  # default far is 2*R
        self.N_samples = N_samples
        self.scene_bounding_sphere = scene_bounding_sphere
        self.take_sphere_intersection = take_sphere_intersection
        
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step

    def get_z_vals(self, ray_dirs, cam_loc, model, REFRACT=True):
        if not self.take_sphere_intersection:
            near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0], 1).cuda()
        else:
            sphere_intersections = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)
            if REFRACT:
                batch_size = 1
                num_pixels = ray_dirs.shape[0]
                sdf = model.implicit_network.get_sdf_vals
                # curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis
                self.far = self.sphere_tracing(batch_size, num_pixels, sdf, cam_loc, ray_dirs, sphere_intersections)
                self.far = self.far.reshape(-1, 1)
                # import ipdb; ipdb.set_trace()
            else:
                self.far = sphere_intersections[:,1:]
            far = self.far
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()
            # far = sphere_intersections[:,1:]

        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()
        z_vals = near * (1. - t_vals) + far * (t_vals)

        if model.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda()

            z_vals = lower + (upper - lower) * t_rand

        return z_vals
    
    def sphere_tracingg(self, batch_size, num_pixels, sdf, cam_loc, ray_directions, sphere_intersections):

        import ipdb; ipdb.set_trace()
        # sphere_intersections[:, 1] : 表示弦和圆相交的另一端

        sphere_intersections_points = cam_loc + sphere_intersections[:, 0].unsqueeze(-1) * ray_directions
        # Initialize start current points
        curr_start_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_start_points = sphere_intersections_points.reshape(-1,3)
        acc_start_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_start_dis = sphere_intersections.reshape(-1,2)[:,0].unsqueeze(-1)

        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start = sdf(curr_start_points)


        while True:
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            curr_sdf_start = next_sdf_start
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            unfinished_mask_start = curr_sdf_start > self.sdf_threshold
            unfinished_mask_start = unfinished_mask_start.reshape(-1)

            if unfinished_mask_start.sum() == 0 or iters == self.sphere_tracing_iters:
                break
            iters += 1
            
            acc_start_dis = acc_start_dis + curr_sdf_start
            
            curr_start_points = (cam_loc + acc_start_dis * ray_directions).reshape(-1, 3)

            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

            not_projected_start = next_sdf_start < 0
            not_proj_iters = 0
            while not_projected_start.sum() > 0 and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_start]

                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start])

                not_projected_start = next_sdf_start < 0
                not_proj_iters += 1

        # curr_start_points, unfinished_mask_start, min_dis先不返回
        # 试试用完整的sphere_tracing? 
        # 找不到 会 一直超 数值就起飞了，一直在iteration
        return acc_start_dis
    
    def sphere_tracing(self, batch_size, num_pixels, sdf, cam_loc, ray_directions, sphere_intersections):
        ''' Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection '''
        # import ipdb; ipdb.set_trace()
        cam_loc = cam_loc[0].reshape(batch_size, 3)
        ray_directions = ray_directions.reshape(batch_size, num_pixels, 3)
        sphere_intersections = sphere_intersections.reshape(batch_size, num_pixels, 2)
        sphere_intersections_points = cam_loc.reshape(batch_size, 1, 1, 3) + sphere_intersections.unsqueeze(-1) * ray_directions.unsqueeze(2) # ray 进 出 bounding_sphere 的交点
        mask_intersect = (sphere_intersections[0][:, 0]>-10).reshape(batch_size, num_pixels)
        unfinished_mask_start = mask_intersect.reshape(-1).clone()
        unfinished_mask_end = mask_intersect.reshape(-1).clone()

        # Initialize start current points
        curr_start_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[:,:,0,:].reshape(-1,3)[unfinished_mask_start]
        acc_start_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1,2)[unfinished_mask_start,0]

        # Initialize end current points
        curr_end_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[:,:,1,:].reshape(-1,3)[unfinished_mask_end]
        acc_end_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1,2)[unfinished_mask_end,1]

        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start]).reshape(-1)

        next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end]).reshape(-1)

        while True:
            # Update sdf
            # import ipdb; ipdb.set_trace()
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            curr_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)
            unfinished_mask_end = unfinished_mask_end & (curr_sdf_end > self.sdf_threshold)

            if (unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end

            # Update points
            curr_start_points = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)
            curr_end_points = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)

            # Fix points which wrongly crossed the surface
            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start]).reshape(-1)

            next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end]).reshape(-1)

            # 如果与单位圆的交点在surface内部, sdf<0, 那么start点后退, end点向前, 相当于从单位圆开始把surface变形
            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_start]

                acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_end[not_projected_end]
                curr_end_points[not_projected_end] = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_end]

                # Calc sdf
                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start]).reshape(-1)
                next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end]).reshape(-1)

                # Update mask
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)
            # unfinished_mask_xx : 表示 ray和单位圆相交了, 但是找不到和surface的交点, 这些需要之后ray_sampler处理, 不在 unfinished_mask_xx 范围内的, 其ray与surface的交点都存在 curr_start_points
            # import ipdb; ipdb.set_trace()
        """
        curr_start_points: 其中大致有三(1, 2.1, 2.2)类点:
            1. 不在mask_sphere里面的, 全存储的0, 这些没有和单位圆(sphere)相交
            2. 在mask_sphere里面的, 这些和单位圆(sphere)相交:
                2.1: 不在 unfinished_mask_xx 里, 也就是 unfinished_mask_xx 存储着False的点: 表示找到了这条ray的和物体相交的点 (sdf == 0), curr_start_points里存储第一个点(start) end 存储着第二个点
                2.2: 在 unfinished_mask_xx 里, 也就是 unfinished_mask_xx 存储着True的点: 表示没有找到这条ray的和物体相交的点 (sdf == 0)
        """
        # return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_disre
        return acc_start_dis
    
class ErrorBoundSampler(RaySampler):
    def __init__(self, scene_bounding_sphere, near, N_samples, N_samples_eval, N_samples_extra,
                 eps, beta_iters, max_total_iters,
                 inverse_sphere_bg=False, N_samples_inverse_sphere=0, add_tiny=0.0,
                 sdf_threshold=5.0e-5, line_search_step=0.5, line_step_iters=1, sphere_tracing_iters=10):
        super().__init__(near, 2.0 * scene_bounding_sphere)
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        
        self.N_samples_extra = N_samples_extra

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny
    
        self.inverse_sphere_bg = inverse_sphere_bg
        if inverse_sphere_bg:
            self.inverse_sphere_sampler = UniformSampler(1.0, 0.0, N_samples_inverse_sphere, False, far=1.0,
                                                         sdf_threshold=sdf_threshold, sphere_tracing_iters=sphere_tracing_iters,
                                                         line_search_step=line_search_step, line_step_iters=line_step_iters)

    def get_z_vals(self, ray_dirs, cam_loc, model):
        ior_0 = 1
        ior_1 = 1.3
        
        cam_loc_list = []
        ray_dirs_list = []
        
        ret_dict = {}
        
        beta0 = model.density.get_beta().detach()

        # Start with uniform sampling
        # 第一个段:
        cam_loc_list.append(cam_loc)
        ray_dirs_list.append(ray_dirs)
        self.uniform_sampler = UniformSampler(self.scene_bounding_sphere, self.near, self.N_samples_eval, take_sphere_intersection=self.inverse_sphere_bg,
                                              sdf_threshold=self.sdf_threshold, sphere_tracing_iters=self.sphere_tracing_iters,
                                              line_search_step=self.line_search_step, line_step_iters=self.line_step_iters)
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs_list[-1], cam_loc_list[-1], model)
        self.far = self.uniform_sampler.far
        z_intersection = self.uniform_sampler.far
        # import ipdb; ipdb.set_trace()
        dists_1 = z_vals[:, 1:] - z_vals[:, :-1]
        bound_1 = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists_1 ** 2.).sum(-1)
        # -------------------- why ...  -------------------- #
        # beta_1 = torch.sqrt(bound_1) # Error！
        beta_1 = bound_1 # OK ! 
        # beta_1 = torch.sqrt(beta_1)
        # -------------------- why ...  -------------------- #

        ret_dict_1 = self.Algorithm(cam_loc_list[-1], ray_dirs_list[-1], model, z_vals, beta_1, beta0)
        # 第一次入射, 改变方向
        intersection = rend_util.get_points(cam_loc=cam_loc_list[-1], ray_dirs=ray_dirs_list[-1], z_vals=z_intersection)
        # points = rend_util.get_points(cam_loc=cam_loc_list[-1], z_vals=ret_dict_1["z_vals"], ray_dirs=ray_dirs_list[-1])
        # points_flat = points.reshape(-1, 3)
        intersection_flat = intersection.reshape(-1, 3)
        gradients = model.implicit_network.gradient(intersection_flat)
        gradients = gradients.detach()
        normals = gradients / gradients.norm(2, -1, keepdim=True)
        normals = normals.reshape(intersection.shape)
        refract_dir, _ = rend_util.refract(ray_in=ray_dirs_list[-1], normal=normals, ior_0=ior_0, ior_1=ior_1)
        
        # 第二个段:
        # import ipdb; ipdb.set_trace()
        cam_loc_list.append(intersection.reshape(-1, 3))
        ray_dirs_list.append(refract_dir)
        self.uniform_sampler = UniformSampler(self.scene_bounding_sphere, self.near, self.N_samples_eval, take_sphere_intersection=self.inverse_sphere_bg,
                                              sdf_threshold=self.sdf_threshold, sphere_tracing_iters=self.sphere_tracing_iters,
                                              line_search_step=self.line_search_step, line_step_iters=self.line_step_iters)
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs_list[-1], cam_loc_list[-1], model)
        self.far = self.uniform_sampler.far
        z_intersection = self.uniform_sampler.far
        dists_2 = z_vals[:, 1:] - z_vals[:, :-1]
        bound_2 = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists_2 ** 2.).sum(-1)
        # beta_2 = torch.sqrt(bound_2)
        beta_2 = bound_2

        ret_dict_2 = self.Algorithm(cam_loc_list[-1], ray_dirs_list[-1], model, z_vals, beta_2, beta0)
        # 第二次入射, 改变方向
        intersection = rend_util.get_points(cam_loc=cam_loc_list[-1], ray_dirs=ray_dirs_list[-1], z_vals=z_intersection)
        # points = rend_util.get_points(cam_loc=cam_loc_list[-1], z_vals=ret_dict_2["z_vals"], ray_dirs=ray_dirs_list[-1])
        # points_flat = points.reshape(-1, 3)
        intersection_flat = intersection.reshape(-1, 3)
        gradients = model.implicit_network.gradient(intersection_flat)
        gradients = gradients.detach()
        normals = gradients / gradients.norm(2, -1, keepdim=True)
        normals = normals.reshape(intersection.shape)
        normals = - normals
        
        refract_dir, mask = rend_util.refract(ray_in=ray_dirs_list[-1], normal=normals, ior_0=ior_0, ior_1=ior_1)
        
        # 第三个段:
        cam_loc_list.append(intersection)
        ray_dirs_list.append(refract_dir)
        self.uniform_sampler = UniformSampler(self.scene_bounding_sphere, self.near, self.N_samples_eval, take_sphere_intersection=self.inverse_sphere_bg,
                                              sdf_threshold=self.sdf_threshold, sphere_tracing_iters=self.sphere_tracing_iters,
                                              line_search_step=self.line_search_step, line_step_iters=self.line_step_iters)
        # import ipdb; ipdb.set_trace()
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs_list[-1][mask.reshape(-1)], cam_loc_list[-1][mask], model, REFRACT=False)
        self.far = self.uniform_sampler.far
        z_intersection = self.uniform_sampler.far
        dists_3 = z_vals[:, 1:] - z_vals[:, :-1]
        bound_3 = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists_3 ** 2.).sum(-1)
        # beta_3 = torch.sqrt(bound_3)
        beta_3 = bound_3

        ret_dict_3 = self.Algorithm(cam_loc_list[-1][mask], ray_dirs_list[-1][mask.reshape(-1)], model, z_vals, beta_3, beta0)
        
        if self.inverse_sphere_bg:
            z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, model)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1./self.scene_bounding_sphere)
            # z_vals = (z_vals, z_vals_inverse_sphere)
    
        ret_dict = {
            "1": {
                "cam_loc": cam_loc_list[0],
                "ray_dirs": ray_dirs_list[0],
                "z_vals": ret_dict_1["z_vals"],
                "z_samples_eik": ret_dict_1["z_samples_eik"]
            },
            "2": {
                "cam_loc": cam_loc_list[1],
                "ray_dirs": ray_dirs_list[1],
                "z_vals": ret_dict_2["z_vals"],
                "z_samples_eik": ret_dict_2["z_samples_eik"]     
            },
            "3": {
                "cam_loc": cam_loc_list[2].reshape(cam_loc_list[1].shape),
                "ray_dirs": ray_dirs_list[2],
                "z_vals": ret_dict_3["z_vals"],
                "z_samples_eik": ret_dict_3["z_samples_eik"]     
            },
            "bg": {
                "z_vals": z_vals_inverse_sphere
            },
            "mask": mask
        }

        return ret_dict

    def Algorithm(self, cam_loc, ray_dirs, model, z_vals, beta, beta0):
        total_iters, not_converge = 0, True
        samples = z_vals
        samples_idx = None
        # Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)

            # Calculating the SDF only for the new sampled points
            with torch.no_grad():
                samples_sdf = model.implicit_network.get_sdf_vals(points_flat)
            if samples_idx is not None:
                sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                       samples_sdf.reshape(-1, samples.shape[1])], -1)
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf


            # Calculating the bound d* (Theorem 1)
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1).cuda()
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign


            # Updating beta using line search
            curr_error = self.get_error_bound(beta0, model, sdf, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.
                curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max


            # Upsample more points
            density = model.density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

            dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)
            free_energy = dists * density
            shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance  # probability of the ray hits something here

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                ''' Sample more points proportional to the current error bound'''

                N = self.N_samples_eval

                bins = z_vals
                # import ipdb; ipdb.set_trace()
                error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * (dists[:,:-1] ** 2.) / (4 * beta.unsqueeze(-1) ** 2)
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (torch.clamp(torch.exp(error_integral),max=1.e6) - 1.0) * transmittance[:,:-1]

                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

            else:
                ''' Sample the final sample set to be used in the volume rendering integral '''

                N = self.N_samples

                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))


            # Invert CDF
            if (not_converge and total_iters < self.max_total_iters) or (not model.training):
                u = torch.linspace(0., 1., steps=N).cuda().unsqueeze(0).repeat(cdf.shape[0], 1)
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N]).cuda()
            u = u.contiguous()

            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = (cdf_g[..., 1] - cdf_g[..., 0])
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])


            # Adding samples if we not converged
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)
                
        # z_vals_ret = z_vals
        z_samples = samples

        # near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0],1).cuda()
        # if self.inverse_sphere_bg: # if inverse sphere then need to add the far sphere intersection
        #     far = rend_util.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:,1:]
        near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()
        far = self.far

        if self.N_samples_extra > 0:
            if model.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1]-1, self.N_samples_extra).long()
            # import ipdb; ipdb.set_trace()
            z_vals_extra = torch.cat([near, far, z_vals[:,sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        # add some of the near surface points
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],)).cuda()
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))
        
        ret_dict = {
            "z_vals": z_vals,
            "z_samples_eik": z_samples_eik
        }
        
        return ret_dict
        
    
    def get_error_bound(self, beta, model, sdf, z_vals, dists, d_star):
        density = model.density(sdf.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), dists * density[:, :-1]], dim=-1)
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists ** 2.) / (4 * beta ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(-integral_estimation[:, :-1])

        return bound_opacity.max(-1)[0]