# !/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021 Hiroaki Santo

import glob
import os

import cv2
import numpy as np
import scipy.io as sio

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

OBJ_NAMES = ["bearPNG", "buddhaPNG", "cowPNG", "pot2PNG", "readingPNG"]

NORMAL_MAP_PNG_FILE_NAME = 'Normal_gt.png'
NORMAL_MAP_TEXT_FILE_NAME = 'normal.txt'
LIGHT_DIRECTIONS_FILE_NAME = 'light_directions.txt'
LIGHT_INTENSITIES_FILE_NAME = 'light_intensities.txt'
MASK_FILE_NAME = 'mask.png'


def load(path: str, view_indices=None, light_indices=None):
    assert os.path.exists(path), "Path does not exist: {}".format(path)
    root_path, obj_name = os.path.split(os.path.normpath(path))

    if view_indices is None:
        view_num = len(sorted(glob.glob(os.path.join(path, "view_*"))))
        view_indices = np.arange(1, view_num + 1)
    else:
        assert min(view_indices) >= 1, "view index starts from 1"

    KK, Rs, Ts = load_calib(path)
    camera_params = {"intrinsics": KK, "Rs": Rs, "Ts": Ts}

    Ms, Ls, masks, Ns = [], [], [], []
    for view_index in tqdm(view_indices):
        view_path = os.path.join(path, "view_{viewindex:02d}".format(viewindex=view_index))

        mask = load_mask(view_path)
        mask[mask != 0] = 1.
        m, n = mask.shape

        M, L = load_measurement(view_path, light_indices=light_indices)
        light_num = len(L)
        M = np.transpose(M, [1, 0, 2]).reshape(light_num, m, n, 3)

        N = load_normal_map(view_path)

        Ms.append(M)
        Ls.append(L)
        masks.append(mask)
        Ns.append(N)

    Ms = np.array(Ms)
    Ls = np.array(Ls)
    masks = np.array(masks)
    Ns = np.array(Ns)

    return Ms, Ls, masks, Ns, camera_params, obj_name


def load_mask(path):
    assert os.path.exists(path), "Path does not exist: {}".format(path)
    mask = cv2.imread(os.path.join(path, MASK_FILE_NAME))[:, :, 0]

    return mask


def load_measurement(path: str, light_indices=None):
    L = load_light_directions(path)
    intensities = load_light_intensities(path)
    light_num, _ = L.shape

    if light_indices is None:
        light_indices = np.arange(1, light_num + 1)
    else:
        assert min(light_indices) >= 1, "light index starts from 1"
        assert max(light_indices) <= light_num, "light index exceeds the number of lights"
        L = L[light_indices]
        intensities = intensities[light_indices]
        light_num = len(light_indices)

    ######
    file_name = '{0:03d}.png'.format(1)
    m_img = cv2.imread(os.path.join(path, file_name))
    m, n, _ = m_img.shape

    ######
    M = np.zeros(shape=(m * n, light_num, 3), dtype=np.float32)
    for l in range(light_num):
        file_name = '{0:03d}.png'.format(light_indices[l])
        m_img = cv2.imread(os.path.join(path, file_name), -1)[:, :, ::-1]
        m_img = m_img.astype(float) / 65536.

        m_img = m_img.reshape(-1, 3)
        M[:, l, :] = m_img

    for l in range(light_num):
        for c in range(3):
            M[:, l, c] /= intensities[l, c]

    return M, L


def load_light_directions(path):
    L = np.loadtxt(os.path.join(path, LIGHT_DIRECTIONS_FILE_NAME))
    return L


def load_light_intensities(path):
    I = np.loadtxt(os.path.join(path, LIGHT_INTENSITIES_FILE_NAME))
    return I


def load_calib(obj_path):
    p = os.path.join(obj_path, "Calib_Results.mat")
    calib = sio.loadmat(p)
    # print(calib.keys())

    KK = calib["KK"]
    view_num = (len(calib.keys()) - 3 - 1) / 2
    assert int(view_num) == view_num, calib.keys()
    Rs = []
    Ts = []
    for view_index in range(1, (len(calib.keys()) - 1) // 2):
        R = calib["Rc_{view_index}".format(view_index=view_index)]
        T = calib["Tc_{view_index}".format(view_index=view_index)]

        Rs.append(R)
        Ts.append(T)

    Rs = np.array(Rs)
    Ts = np.array(Ts)

    return KK, Rs, Ts


def load_normal_map(path):
    p = os.path.join(path, "Normal_gt.mat")
    normal_map = sio.loadmat(p)["Normal_gt"]
    return normal_map

