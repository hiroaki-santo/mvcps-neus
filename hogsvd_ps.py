#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Hiroaki Santo

import multiprocessing as mp
import os

import cv2
import joblib
import numpy as np

from hogsvd_rank_def import hogsvd


def sample_valid_pixels(X, mask, max_sampling_pixel_num: int = 0, seed=None):
    data_num, pixel_num, ch = X.shape
    assert mask.shape == (data_num, pixel_num), (X.shape, mask.shape)

    rand = np.random.RandomState(seed)

    valid_pixel_nums = np.sum(mask, axis=-1)
    sample_pixel_num = min(valid_pixel_nums)
    if max_sampling_pixel_num is not None and max_sampling_pixel_num > 0:
        sample_pixel_num = min(sample_pixel_num, max_sampling_pixel_num)

    sampled = []
    sampled_idx = []
    for i in range(data_num):
        idx = np.arange(pixel_num)[mask[i] != 0]
        idx = idx[rand.permutation(len(idx))[:sample_pixel_num]]
        sampled.append(X[i][idx])
        sampled_idx.append(idx)
    sampled = np.stack(sampled, axis=0)
    sampled_idx = np.stack(sampled_idx, axis=0)
    assert sampled.shape == (data_num, sample_pixel_num, ch), sampled.shape
    assert sampled_idx.shape == (data_num, sample_pixel_num), sampled_idx.shape
    return sampled, sampled_idx


def hosvd_ransac(M, mask, ransac_pixel_num: int, seed=None):
    view_num, pixel_num, light_num = M.shape
    assert mask.shape == (view_num, pixel_num), (M.shape, mask.shape)

    X, sample_indices = sample_valid_pixels(M, mask, max_sampling_pixel_num=ransac_pixel_num, seed=seed)

    U, S, V, Tau, taumin, taumax, iso_classes = hogsvd(X.reshape(-1, light_num),
                                                       m=[X[i].shape[0] for i in range(X.shape[0])])
    us = U.reshape(X.shape[0], -1, X.shape[-1])
    ss = S.reshape(X.shape[0], -1, X.shape[-1])
    L_ = V.T[-3:, :].T  # (light, 3)

    # (view, pixel, light), (view, light, light), (light, light)
    X_recover = us @ ss @ V.T  # view, pixel, light
    residual = np.linalg.norm(X - X_recover)

    Ns_ = np.einsum("lk,vpl->vpk", np.linalg.pinv(L_).T, M)
    res_map = np.mean((Ns_ @ L_.T - M) ** 2, axis=-1)
    assert res_map.shape == mask.shape, (res_map.shape, mask.shape)

    mask = np.where(mask != 0, 1, 0)
    res_map = res_map * mask
    err = np.abs(res_map[mask != 0]).sum() / mask.sum()

    return residual, err, L_, seed


def run_hogsvd(M, mask, ransac_iter: int = 200, ransac_sampling_num: int = 5000,
               cpu_count: int = mp.cpu_count() - 1):
    view_num, light_num, m, n, ch = M.shape
    assert ch == 3 or ch == 1, M.shape
    assert mask.shape == (view_num, m, n), mask.shape

    M = np.mean(M.astype(np.float32), axis=-1)
    M = M.reshape([view_num, light_num, m * n]).transpose([0, 2, 1])  # (view, pixel, light)
    mask = mask.reshape([view_num, m * n])

    ransac_results = joblib.Parallel(n_jobs=cpu_count, verbose=1)(
        joblib.delayed(hosvd_ransac)(M, mask, ransac_pixel_num=ransac_sampling_num, seed=seed)
        for seed in range(ransac_iter))

    best_data = min(ransac_results, key=lambda x: x[1])
    residual, err, L_, seed = best_data
    print("best_data", best_data)

    Ns_ = np.einsum("lk,vpl->vpk", np.linalg.pinv(L_).T, M)
    res_map = np.mean((Ns_ @ L_.T - M) ** 2, axis=-1)

    Ns_ = Ns_.reshape([view_num, m, n, 3])
    res_map = res_map.reshape([view_num, m, n])
    mask = mask.reshape([view_num, m, n])

    return {"Lest": L_, "Nest": Ns_, "residual_map": res_map}



def main(cpu_count: int = max(1, mp.cpu_count() - 1)):
    diligent_dataset_path = "data/DiLiGenT-MV/mvpmsData"
    output_root_path = "data/mvcps/diligentmv/"
    obj_name_list = ["bearPNG", "buddhaPNG", "cowPNG", "pot2PNG", "readingPNG"]

    view_sampling_num = 4
    view_sampling_offset = 0
    light_sampling_num = 3
    light_sampling_seed = 12

    for obj_name in obj_name_list:
        print(obj_name)

        from diligentmv import load
        M, Lgt, mask, Ngt, camera_params, obj_name = load(os.path.join(diligent_dataset_path, obj_name))
        view_num, light_num, m, n, _ = M.shape
        assert Lgt.shape == (view_num, light_num, 3), Lgt.shape
        assert mask.shape == (view_num, m, n), mask.shape
        assert Ngt.shape == (view_num, m, n, 3), Ngt.shape
        Lgt = np.mean(Lgt, axis=0)
        Lgt = Lgt / np.linalg.norm(Lgt, axis=-1, keepdims=True)
        mask[mask != 0] = 1

        view_sampling_offset = view_sampling_offset % (view_num // view_sampling_num)
        view_indices = np.arange(view_num)[view_sampling_offset::view_num // view_sampling_num][:view_sampling_num]

        M = M[view_indices]
        mask = mask[view_indices]
        Ngt = Ngt[view_indices]
        view_num, light_num, m, n, _ = M.shape

        output_dir_path = os.path.join(output_root_path, obj_name)

        print(obj_name, f"v{view_sampling_num}o{view_sampling_offset}l{light_sampling_num}s{light_sampling_seed}")

        if light_sampling_num > 0:
            rand = np.random.RandomState(light_sampling_seed)
            light_indices = sorted(rand.permutation(light_num)[:light_sampling_num])
        else:
            light_sampling_seed = 0
            light_sampling_num = light_num
            light_indices = np.arange(light_num)

        # check rank of Light matrix
        print(Lgt.shape, light_indices)
        rank = np.linalg.matrix_rank(Lgt[light_indices], tol=0.1)
        if rank < 3:
            raise ValueError(f"rank of Lgt is {rank} < 3. Please change the random seed for light sampling.")

        est_dict = run_hogsvd(M=M[:, light_indices], mask=mask,
                              ransac_iter=1, ransac_sampling_num=999999999,  # no ransac
                              cpu_count=cpu_count)
        Nest = est_dict["Nest"]

        opath = os.path.join(output_dir_path,
                             f"v{view_sampling_num}o{view_sampling_offset}l{light_sampling_num}s{light_sampling_seed}")
        os.makedirs(opath, exist_ok=True)
        np.savez(os.path.join(opath, "estimated.npz"), Lest=est_dict["Lest"], Nest=est_dict["Nest"])

        Nest = Nest / (np.linalg.norm(Nest, axis=-1, keepdims=True) + 1e-6)
        for vidx, v in enumerate(view_indices):
            n_img = (Nest[vidx, :, :, :] + 1) / 2.
            cv2.imwrite(os.path.join(opath, f"v{v:08d}.png"),
                        (n_img * np.iinfo(np.uint16).max).astype(np.uint16)[:, :, ::-1])


if __name__ == '__main__':
    main()
