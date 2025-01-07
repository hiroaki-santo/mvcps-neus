import os
from glob import glob

import cv2 as cv
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        self.light_sampling_num = conf.get_int('light_sampling_num')
        self.light_sampling_seed = conf.get_int('light_sampling_seed')
        self.view_sampling_num = conf.get_int('view_sampling_num')
        self.view_sampling_offset = conf.get_int('view_sampling_offset')

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        paths = os.path.join(self.data_dir, "image", "*.png")
        images_list = sorted(glob(paths))
        self.n_images = len(images_list)
        assert self.n_images > 0, f'No images found in {paths}'

        images_np = np.stack([cv.imread(im_name)[:, :, ::-1] for im_name in images_list]) / 255.
        masks_list = sorted(glob(os.path.join(self.data_dir, "mask", "*.png")))
        assert len(masks_list) == self.n_images, masks_list
        masks_np = np.stack([cv.imread(im_name, cv.IMREAD_GRAYSCALE)[..., None] for im_name in masks_list])
        masks_np[masks_np != 0] = 1.  # [B, H, W, 1]

        world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        intrinsics_all = []
        pose_all = []

        for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks = torch.from_numpy(masks_np.astype(np.float32)).cpu()  # [n_images, H, W, 1]
        self.intrinsics_all = torch.stack(intrinsics_all).to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        ##################################################
        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])

        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        self.normal_input_confidence_map = torch.ones(self.n_images, self.H, self.W, 1).cpu()

        self.use_normal = True
        normal_gt_local_dir_name = "normal_local"
        if not os.path.exists(os.path.join(self.data_dir, normal_gt_local_dir_name)):
            print("No GT normal map found, use dummy normal map")
            normal_gt_local_np = np.ones((self.n_images, self.H, self.W, 3))
            normal_gt_local_np[..., [0, 1]] = 0
        else:
            base_path = os.path.join(self.data_dir, normal_gt_local_dir_name, '*.png')
            normal_gt_local_list = sorted(glob(base_path))
            assert len(normal_gt_local_list) > 0, base_path
            assert len(normal_gt_local_list) == self.n_images, (
                normal_gt_local_list, len(normal_gt_local_list), self.n_images)

            normal_gt_local_np = np.stack([cv.imread(p)[:, :, ::-1] for p in normal_gt_local_list]).astype(
                float) / 255. * 2 - 1
            normal_gt_local_np = normal_gt_local_np / (
                    np.linalg.norm(normal_gt_local_np, axis=-1, keepdims=True) + 1e-6)
            normal_gt_local_np[..., 0] *= 1
            normal_gt_local_np[..., 1] *= -1
            normal_gt_local_np[..., 2] *= -1

        self.normal_gt_local = torch.from_numpy(normal_gt_local_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]

        ##################################################
        self.view_sampling_offset = self.view_sampling_offset % (self.n_images // self.view_sampling_num)
        view_indices = np.arange(self.n_images)[
                       self.view_sampling_offset::self.n_images // self.view_sampling_num][:self.view_sampling_num]

        normal_input_root_path = conf.get_string("normal_input_root_path", self.data_dir)
        normal_input_local_dir_name = f"v{self.view_sampling_num}o{self.view_sampling_offset}l{self.light_sampling_num}s{self.light_sampling_seed}"
        normal_input_local_np = \
            np.load(os.path.join(normal_input_root_path, normal_input_local_dir_name, "estimated.npz"))['Nest']

        normal_input_local_np = normal_input_local_np / np.abs(normal_input_local_np).max()
        normal_input_local_np[..., 0] *= 1
        normal_input_local_np[..., 1] *= -1
        normal_input_local_np[..., 2] *= -1

        self.normal_input_local = torch.zeros(size=(self.n_images, self.H, self.W, 3), dtype=torch.float32).cpu()
        self.normal_input_local[view_indices] = torch.from_numpy(normal_input_local_np.astype(np.float32)).cpu()

        assert self.normal_gt_local.shape == (self.n_images, self.H, self.W, 3), self.normal_gt_local.shape
        assert self.normal_input_local.shape == (self.n_images, self.H, self.W, 3), self.normal_input_local.shape

        self.use_depth = False
        self.depth_input_local = torch.zeros((self.n_images, self.H, self.W, 1), dtype=torch.float32)
        assert self.depth_input_local.shape == (self.n_images, self.H, self.W, 1), self.depth_input_local.shape

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)

        W, H = pixels_x.shape
        pixels_x = pixels_x.reshape(-1)
        pixels_y = pixels_y.reshape(-1)

        rays = self._rays_at(img_idx, pixels_x, pixels_y)
        assert rays.shape == (W * H, 3 + 3 + 3 + 1 + 3 + 3 + 1 + 1 + 2), (rays.shape, W, H)
        return rays.reshape(W, H, -1).transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).cpu()
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).cpu()
        return self._rays_at(img_idx, pixels_x, pixels_y)

    def _rays_at(self, img_idx, pixels_x, pixels_y):
        pixels_x = pixels_x.cpu().long()
        pixels_y = pixels_y.cpu().long()

        color = self.images[img_idx][(pixels_y, pixels_x)].to(self.device)  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)].to(self.device)  # batch_size, 1
        normal_gt = self.normal_gt_local[img_idx][(pixels_y, pixels_x)].to(self.device)  # batch_size, 3
        normal_input = self.normal_input_local[img_idx][(pixels_y, pixels_x)].to(self.device)  # batch_size, 3
        normal_input_confidence = self.normal_input_confidence_map[img_idx][(pixels_y, pixels_x)].to(
            self.device)  # batch_size, 1
        depth_input = self.depth_input_local[img_idx][(pixels_y, pixels_x)].to(self.device)  # batch_size, 1

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1, ).to(torch.float32).to(self.device)
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat(
            [rays_o, rays_v, color, mask[:, :1], normal_gt, normal_input, normal_input_confidence, depth_input,
             torch.stack([pixels_x, pixels_y], dim=-1).to(torch.float32).to(self.device)], dim=-1)

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_list[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

