import argparse
import logging
import os
import random
from collections import OrderedDict
from shutil import copyfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pyhocon import ConfigFactory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer


class Runner:
    def __init__(self, conf_path, case='CASE_NAME', is_continue=False, run_id: str = ""):
        self.device = torch.device('cuda')

        self.conf_path = conf_path
        with open(self.conf_path, 'r') as f:
            conf_text = f.read()
            conf_text = conf_text.replace('CASE_NAME', case)
        self.conf = ConfigFactory.parse_string(conf_text)

        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        if len(run_id) > 0:
            self.base_exp_dir = os.path.join(self.base_exp_dir, run_id)
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset = Dataset(self.conf['dataset'])

        self.iter_step = 0

        self.view_sampling = self.conf.get_int('dataset.view_sampling_num')
        self.view_sampling_offset = self.conf.get_int('dataset.view_sampling_offset')

        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.learning_rate_ambiguity_matrix = self.conf.get_float('train.learning_rate_ambiguity_matrix', 0.0)
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        #
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.use_confidence_start = self.conf.get_float('train.use_confidence_start', default=self.warm_up_end)
        self.smoothed_pe_end = self.conf.get_float('train.smoothed_pe_end', default=self.warm_up_end)
        self.warm_up_normal_loss_end = self.conf.get_float('train.warm_up_normal_loss_end', default=self.warm_up_end)

        assert self.warm_up_end >= 0, self.warm_up_end
        assert self.anneal_end >= 0, self.anneal_end
        assert self.use_confidence_start >= 0, self.use_confidence_start
        assert self.smoothed_pe_end >= 0, self.smoothed_pe_end
        assert self.warm_up_normal_loss_end >= 0, self.warm_up_normal_loss_end

        if self.warm_up_end <= 1.0:
            self.warm_up_end = int(self.warm_up_end * self.end_iter)
        if self.anneal_end <= 1.0:
            self.anneal_end = int(self.anneal_end * self.end_iter)
        if self.use_confidence_start <= 1.0:
            self.use_confidence_start = int(self.use_confidence_start * self.end_iter)
        if self.smoothed_pe_end <= 1.0:
            self.smoothed_pe_end = int(self.smoothed_pe_end * self.end_iter)
        if self.warm_up_normal_loss_end <= 1.0:
            self.warm_up_normal_loss_end = int(self.warm_up_normal_loss_end * self.end_iter)

        # Weights
        self.color_weight = self.conf.get_float('train.color_weight', 1.0)
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.bias_loss_weight = self.conf.get_float("train.bias_loss_weight", 0.)
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.normal_weight = self.conf.get_float('train.normal_weight', 0.0)
        self.depth_weight = self.conf.get_float('train.depth_weight', 0.0)
        self.use_normal_confidence = self.conf.get_bool('train.use_normal_confidence', False)

        self.is_continue = is_continue
        self.mode = "train"
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)

        self.normal_ambiguity_matrix = torch.nn.Parameter(torch.rand((3, 3), device=self.device) * 2. - 1)

        self.normal_confidence_data = OrderedDict()

        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        self.optimizer_ambiguity_matrix = torch.optim.Adam([self.normal_ambiguity_matrix],
                                                           lr=self.learning_rate_ambiguity_matrix)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

    @classmethod
    def disambiguate_normal(cls, ambiguity_matrix, input_normal_local):
        if len(input_normal_local.shape) == 2:  # B x 3
            input_normal_local = torch.einsum('ij,bi->bj', ambiguity_matrix, input_normal_local)
        elif len(input_normal_local.shape) == 3:  # H x W x 3
            input_normal_local = torch.einsum('ij,hwi->hwj', ambiguity_matrix, input_normal_local)
        input_normal_local = F.normalize(input_normal_local, dim=-1, p=2)

        return input_normal_local


    def compute_normal_loss(self, image_idx, input_normal_local, mask, render_out):

        normals_world = render_out['normal']

        rot = self.dataset.pose_all[image_idx, :3, :3]

        normals_world = F.normalize(normals_world, dim=-1, p=2)
        normals_local = torch.einsum('ij,bi->bj', rot, normals_world)

        input_normal_local = self.disambiguate_normal(self.normal_ambiguity_matrix, input_normal_local)

        normal_l1_loss = torch.abs(normals_local - input_normal_local)

        normal_l1_loss = normal_l1_loss * mask
        normal_l1_loss = normal_l1_loss.mean(dim=-1)
        assert normal_l1_loss.shape == (len(mask),), (normal_l1_loss.shape, mask.shape)

        normal_ang_error = torch.acos(
            (F.normalize(normals_local, dim=-1, p=2) * F.normalize(input_normal_local, dim=-1, p=2)).sum(dim=-1).clip(
                -1, 1))
        normal_ang_error = normal_ang_error.flatten() * mask.flatten()
        assert normal_ang_error.shape == (len(mask),), normal_ang_error.shape

        return normal_l1_loss, normal_ang_error

    def prepare_data(self, data):
        rays_o = data[:, :3]
        rays_d = data[:, 3:6]
        true_rgb = data[:, 6:9]
        mask = data[:, 9:10]
        input_normal_local = data[:, 13:16]
        normal_confidence = data[:, 16:17]
        return rays_o, rays_d, true_rgb, mask, input_normal_local, normal_confidence

    def compute_loss(self, rays_o, rays_d, true_rgb, mask, input_normal_local, normal_confidence, image_idx):
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

        mask = (mask > 0.5).float() if self.mask_weight > 0.0 else torch.ones_like(mask)
        mask_sum = mask.sum() + 1e-5

        render_out = self.renderer.render(rays_o, rays_d, near, far,
                                          background_rgb=background_rgb,
                                          cos_anneal_ratio=self.get_cos_anneal_ratio())

        color_fine = render_out['color_fine']
        gradient_error = render_out['gradient_error']
        weight_sum = render_out['weight_sum']

        color_error = (color_fine - true_rgb) * mask
        color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        psnr = 20.0 * torch.log10(1.0 / ((color_error ** 2).sum() / (mask_sum * 3.0)).sqrt())

        eikonal_loss = gradient_error

        mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

        if self.dataset.use_normal:
            normal_loss, normal_ang_error = self.compute_normal_loss(
                image_idx=image_idx, mask=mask, input_normal_local=input_normal_local, render_out=render_out
            )
            if self.use_normal_confidence:
                normal_loss = normal_loss[..., None] * normal_confidence
            normal_loss = normal_loss.sum() / mask_sum
            normal_ang_error = normal_ang_error.sum() / mask_sum
        else:
            normal_loss = torch.zeros(1, device=self.device)
            normal_ang_error = torch.zeros(1, device=self.device)

        return {
            'render_out': render_out,
            'color_fine_loss': color_fine_loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
            'normal_loss': normal_loss,
            'psnr': psnr,
            'normal_ang_error': normal_ang_error,
            'mask_sum': mask_sum
        }

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            image_idx = image_perm[self.iter_step % len(image_perm)]

            self.nerf_outside.update_training_progress(iter_i, start=0, end=self.smoothed_pe_end)
            self.sdf_network.update_training_progress(iter_i, start=0, end=self.smoothed_pe_end)
            self.color_network.update_training_progress(iter_i, start=0, end=self.smoothed_pe_end)

            data = self.dataset.gen_random_rays_at(image_idx, self.batch_size)
            rays_o, rays_d, true_rgb, mask, input_normal_local, normal_confidence = self.prepare_data(data)

            losses = self.compute_loss(rays_o, rays_d, true_rgb, mask, input_normal_local, normal_confidence, image_idx)
            color_fine_loss = losses['color_fine_loss']
            eikonal_loss = losses['eikonal_loss']
            mask_loss = losses['mask_loss']
            normal_loss = losses['normal_loss']

            if self.normal_weight > 0:
                alpha1 = 0.03 * self.normal_weight
                alpha2 = self.normal_weight

                start_i = self.warm_up_end
                end_i = self.warm_up_normal_loss_end
                progress = np.clip((self.iter_step - start_i) / (end_i - start_i + 1e-6), 0, 1.)
                normal_weight = alpha1 + progress * (alpha2 - alpha1)
            else:
                normal_weight = 0.0

            loss = color_fine_loss * self.color_weight + \
                   eikonal_loss * self.igr_weight + \
                   mask_loss * self.mask_weight + \
                   normal_loss * normal_weight
            self.optimizer.zero_grad()
            self.optimizer_ambiguity_matrix.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_ambiguity_matrix.step()
            self.iter_step += 1

            if self.use_normal_confidence:
                if self.iter_step > self.smoothed_pe_end:
                    if self.iter_step % 2000 == 0:
                        self.update_input_normal_confidence()
                else:
                    if self.iter_step % 500 == 0:
                        self.update_input_normal_confidence()



            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print(image_perm)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))
                print("color_fine_loss", color_fine_loss.item(), "*", self.color_weight)
                print("eikonal_loss", eikonal_loss.item(), "*", self.igr_weight)
                print("mask_loss", mask_loss.item(), "*", self.mask_weight)
                print("normal_loss", normal_loss.item(), "*", normal_weight)
                print(self.normal_ambiguity_matrix)

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(resolution=768)

            if self.iter_step % self.val_freq == 0:
                if len(self.get_image_perm()) == self.dataset.n_images:
                    self.validate_image()
                else:
                    if (self.iter_step // self.val_freq) % 2 == 0:
                        self.validate_image(idx=self.get_image_perm()[0])
                    else:
                        valid_idx = [v for v in range(self.dataset.n_images) if v not in self.get_image_perm()]
                        self.validate_image(idx=valid_idx[(self.iter_step // self.val_freq) % len(valid_idx)])

            if self.iter_step % (self.save_freq * 10) == 0:
                MAngEs = []
                for i in range(self.dataset.n_images):
                    MAngE = self.validate_image(idx=i, resolution_level=1)
                    MAngEs.append(MAngE)
                print(f"{self.iter_step}: MAngE for {self.dataset.n_images} views", np.mean(MAngEs))

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def update_input_normal_confidence(self):
        stacked_diff_maps = self.compute_response_to_projection_opt(resolution_level=3)
        self.normal_confidence_data[self.iter_step] = stacked_diff_maps

        if self.iter_step < self.use_confidence_start:
            self.dataset.normal_input_confidence_map[:] = 1.
            return

        data = {}
        for i, idx in enumerate(sorted(self.get_image_perm())):
            data[idx] = []
            for iter_step, stacked_diff_maps in self.normal_confidence_data.items():
                if iter_step < self.use_confidence_start:
                    continue

                diff_stacked = stacked_diff_maps[idx][[t for t in range(len(self.get_image_perm())) if t != i]]
                assert len(diff_stacked.shape) == 3, diff_stacked.shape

                x = diff_stacked[:, None]
                resized = F.interpolate(x, size=(self.dataset.H, self.dataset.W), mode='nearest')[:, 0]
                data[idx].append(resized)

            data[idx] = torch.cat(data[idx], dim=0)

        mean = {k: torch.mean(v, dim=0) for k, v in data.items()}
        std = {k: torch.std(v, dim=0) for k, v in data.items()}

        confidence = torch.zeros_like(self.dataset.normal_input_confidence_map)
        for idx in sorted(self.get_image_perm()):
            tmp = std[idx]
            tmp = tmp / tmp.max()
            f = lambda x: torch.clip(torch.exp(-5 * x), 0, 1)
            confidence[idx] = f(tmp)[..., None]

        self.dataset.normal_input_confidence_map = confidence

    def compute_response_to_projection_opt(self, resolution_level: int = 1):

        data_dict = {}
        prev_dict = {}
        for idx in sorted(self.get_image_perm()):
            data_dict[idx] = []
            #
            rays = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
            rays_o, rays_d, _, mask_gt, _, normal_input, _, _, _ = torch.split(
                rays, [3, 3, 3, 1, 3, 3, 1, 1, 2], dim=-1)
            H, W, _ = rays_o.shape
            _, sdf_normal_world = self._render_rays(rays_o, rays_d)
            sdf_normal_world = sdf_normal_world.detach().cpu()
            rot = self.dataset.pose_all[idx, :3, :3].detach().cpu()
            mask_gt = mask_gt.detach().cpu()
            sdf_normal_local = torch.einsum("ij,kli->klj", rot, sdf_normal_world.detach().cpu())
            assert sdf_normal_local.shape == (H, W, 3), (sdf_normal_local.shape, H, W)

            prev_dict[idx] = {"sdf_normal_local": sdf_normal_local.detach().cpu(),
                              "normal_input": normal_input.detach().cpu(),
                              "mask": mask_gt.detach().cpu()}

        for image_idx in sorted(self.get_image_perm()):
            data = self.dataset.gen_random_rays_at(image_idx, self.batch_size)
            rays_o, rays_d, true_rgb, mask, input_normal_local, normal_confidence = self.prepare_data(data)

            losses = self.compute_loss(rays_o, rays_d, true_rgb, mask, input_normal_local, normal_confidence, image_idx)
            color_fine_loss = losses['color_fine_loss']
            eikonal_loss = losses['eikonal_loss']
            mask_loss = losses['mask_loss']
            normal_loss = losses['normal_loss']

            loss = color_fine_loss * self.color_weight + \
                   eikonal_loss * self.igr_weight + \
                   mask_loss * self.mask_weight + \
                   normal_loss * self.normal_weight

            self.optimizer.zero_grad()
            self.optimizer_ambiguity_matrix.zero_grad()
            loss.backward()

            updated_ambiguity_matrix = self.normal_ambiguity_matrix - self.normal_ambiguity_matrix.grad * \
                                       self.optimizer_ambiguity_matrix.param_groups[0]['lr']
            updated_ambiguity_matrix = updated_ambiguity_matrix.detach().cpu()

            ###########################################################

            for image_idx_eval in self.get_image_perm():
                image_idx_eval = int(image_idx_eval)

                sdf_normal_local = prev_dict[int(image_idx_eval)]["sdf_normal_local"]
                normal_input = prev_dict[int(image_idx_eval)]["normal_input"]
                mask = prev_dict[int(image_idx_eval)]["mask"]

                projected_normal_prev = self.disambiguate_normal(self.normal_ambiguity_matrix.detach().cpu(),
                                                                 normal_input)
                projected_normal_after = self.disambiguate_normal(updated_ambiguity_matrix, normal_input)

                def __l2_error(a, b):
                    assert a.shape == b.shape, (a.shape, b.shape)
                    return torch.mean((a - b) ** 2, dim=-1)

                diff_map = __l2_error(sdf_normal_local, projected_normal_after) - \
                           __l2_error(sdf_normal_local, projected_normal_prev)

                diff_map[mask.squeeze(-1) == 0] = 0
                data_dict[int(image_idx_eval)].append(diff_map.detach())
            ###########################################################

        stacked_diff_maps = {}
        for image_idx in sorted(self.get_image_perm()):
            stacked_diff_maps[int(image_idx)] = torch.stack(data_dict[int(image_idx)])

        return stacked_diff_maps

    def get_image_perm(self):
        if self.view_sampling <= 0:
            indices = np.arange(self.dataset.n_images)
        else:
            assert self.dataset.n_images >= self.view_sampling, (self.dataset.n_images, self.view_sampling)
            view_sampling_offset = self.view_sampling_offset % (self.dataset.n_images // self.view_sampling)
            indices = np.arange(self.dataset.n_images)[
                      view_sampling_offset::self.dataset.n_images // self.view_sampling][:self.view_sampling]
        random.shuffle(indices)
        return indices

    def get_random_neighbour_image_idx(self, image_idx, seed=None):
        train_indices = sorted(self.get_image_perm())
        assert image_idx in train_indices, (image_idx, train_indices)

        n = len(train_indices)
        idx = train_indices.index(image_idx)
        prev_i = train_indices[(idx - 1) % n]
        next_i = train_indices[(idx + 1) % n]

        rand = np.random.RandomState(seed)
        return rand.choice([prev_i, next_i])

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

        for g in self.optimizer_ambiguity_matrix.param_groups:
            g['lr'] = self.learning_rate_ambiguity_matrix * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer_ambiguity_matrix.load_state_dict(checkpoint['optimizer_ambiguity_matrix'])
        self.optimizer_ambiguity_matrix.data = checkpoint['normal_ambiguity_matrix']
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'optimizer_ambiguity_matrix': self.optimizer_ambiguity_matrix.state_dict(),
            'normal_ambiguity_matrix': self.normal_ambiguity_matrix.detach().cpu(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def _render_rays(self, rays_o, rays_d):
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu())

            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu()
                out_normal_fine.append(normals)
            del render_out

        color_rendered = torch.cat(out_rgb_fine, dim=0).reshape([H, W, 3])
        sdf_normal_world = torch.cat(out_normal_fine, dim=0).reshape([H, W, 3])
        sdf_normal_world = F.normalize(sdf_normal_world, dim=-1, eps=1e-6)

        return color_rendered, sdf_normal_world

    def validate_image(self, idx=-1, resolution_level=-1, tag: str = ""):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        rays_o, rays_d, color_gt, mask_gt, normal_gt, normal_input, normal_confidence, depth_input, pixel_coordinates = \
            torch.split(rays, [3, 3, 3, 1, 3, 3, 1, 1, 2], dim=-1)
        H, W, _ = rays_o.shape
        color_rendered, sdf_normal_world = self._render_rays(rays_o, rays_d)

        color_rendered = color_rendered.detach().cpu()
        sdf_normal_world = sdf_normal_world.detach().cpu()

        rot = self.dataset.pose_all[idx, :3, :3].detach().cpu()
        mask_gt = mask_gt.detach().cpu()
        normal_gt = normal_gt.detach().cpu()
        sdf_normal_local = torch.einsum("ij,kli->klj", rot, sdf_normal_world.detach().cpu())
        assert sdf_normal_local.shape == (H, W, 3), (sdf_normal_local.shape, H, W)

        projected_normal = self.disambiguate_normal(self.normal_ambiguity_matrix.detach().cpu(),
                                                    normal_input.detach().cpu())

        normal_ang_error_sdf = \
            torch.acos((F.normalize(sdf_normal_local, dim=-1, eps=1e-6) *
                        F.normalize(normal_gt, dim=-1, eps=1e-6)).sum(dim=2).clip(-1, 1.))[..., None]
        normal_ang_error_sdf = normal_ang_error_sdf.detach().cpu()
        assert normal_ang_error_sdf.shape == (H, W, 1), (normal_ang_error_sdf.shape, H, W)

        output_dir_color = os.path.join(self.base_exp_dir, 'eval_color')
        output_dir_normal = os.path.join(self.base_exp_dir, 'eval_normal')

        os.makedirs(output_dir_color, exist_ok=True)
        os.makedirs(output_dir_normal, exist_ok=True)

        cv2.imwrite(os.path.join(output_dir_color, f'{self.iter_step:0>8d}_{idx:08d}{tag}.png'),
                    (np.concatenate([
                        color_rendered.detach().cpu().numpy(),
                        color_gt.detach().cpu().numpy()
                    ]) * 255.).astype(np.uint8)[:, :, ::-1])

        cv2.imwrite(os.path.join(output_dir_normal, f'{self.iter_step:0>8d}_{idx:08d}{tag}.png'),
                    (np.concatenate([
                        sdf_normal_local.detach().cpu().numpy(),
                        projected_normal.detach().cpu().numpy(),
                        normal_gt.detach().cpu().numpy(),
                    ]) * 128 + 128).astype(np.uint8)[:, :, ::-1])

        return torch.rad2deg(torch.mean(normal_ang_error_sdf[mask_gt != 0]))

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, required=True)
    parser.add_argument('--run_id', type=str, default='')

    args = parser.parse_args()


    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.case, args.is_continue, run_id=args.run_id)
    runner.train()
