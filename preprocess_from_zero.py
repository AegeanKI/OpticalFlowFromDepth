import os
import sys
import time
import math
import glob

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

import utils
import geometry
from bilateral_filter import sparse_bilateral_filtering
from my_cuda.fw import FW
from collections import defaultdict
import random
from argparse import ArgumentParser
from dataloader import ReDWeb, DIML, FiltedReDWeb
import copy

class SpecialFlow(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

    def forward(self, size, augment_flow_type):
        if augment_flow_type >= 7.:
            p1, p_prev = self._shear(size)
        elif augment_flow_type >= 6.:
            p1, p_prev = self._rotate(size)
        elif augment_flow_type >= 5.:
            p1, p_prev = self._flip(size)

        p0 = self._get_p0(size)

        special_flow = (p1 - p0).permute(2, 0, 1).squeeze(0)
        back_special_flow = (p_prev - p0).permute(2, 0, 1).squeeze(0)

        del p1, p_prev
        return special_flow, back_special_flow

    def _flip(self, size):
        h, w = size
        horizontal = utils.get_random(0, 1) > 0

        if horizontal:
            meshgrid = torch.meshgrid(torch.arange(w - 1, -1, -1), torch.arange(h), indexing="xy")
        else:
            meshgrid = torch.meshgrid(torch.arange(w), torch.arange(h - 1, -1, -1), indexing="xy")

        p1 = torch.stack(meshgrid, axis=-1).type(torch.float32).to(self.device)
        p_prev = p1

        del horizontal, meshgrid
        return p1, p_prev

    def _rotate(self, size):
        h, w = size
        c0 = (utils.get_random(w / 4, w / 2) + w / 2, utils.get_random(h / 4, h / 2) + h / 2)
        c0 = torch.tensor(c0).to(self.device)
        theta = torch.deg2rad(utils.get_random(2, 8)).to(self.device) # 8 - 10
        # theta = torch.deg2rad(utils.get_random(3, 5)).to(self.device) # 5 - 8

        rotate = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]]).type(torch.float32).to(self.device)
        reverse_rotate = torch.tensor([[torch.cos(-theta), -torch.sin(-theta)],
                                       [torch.sin(-theta), torch.cos(-theta)]]).type(torch.float32).to(self.device)

        p0 = self._get_p0(size)
        p1 = (p0 - c0) @ rotate + c0
        p_prev = (p0 - c0) @ reverse_rotate + c0

        del c0, theta
        del rotate, reverse_rotate
        return p1, p_prev

    def _shear(self, size):
        h, w = size
        horizontal = utils.get_random(0, 1) > 0
        shear_range = utils.get_random(0.15, 0.2) # 0.2 - 0.35
        # shear_range = utils.get_random(0.2, 0.1) # 0.1 - 0.3

        if horizontal:
            shear = torch.tensor([[1, 0], [shear_range, 1]]).type(torch.float32).to(self.device)
            reverse_shear = torch.tensor([[1, 0], [-shear_range, 1]]).type(torch.float32).to(self.device)
        else:
            shear = torch.tensor([[1, shear_range], [0, 1]]).type(torch.float32).to(self.device)
            reverse_shear = torch.tensor([[1, -shear_range], [0, 1]]).type(torch.float32).to(self.device)
        
        p0 = self._get_p0(size)
        p1 = p0 @ shear
        p_prev = p0 @ reverse_shear

        del horizontal, shear_range
        del shear, reverse_shear
        return p1, p_prev

    def _get_p0(self, size):
        h, w = size
        meshgrid = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        p0 = torch.stack(meshgrid, axis=-1).type(torch.float32).to(self.device)
        return p0


def augment_flow_small(img0, img0_depth, img1, img1_depth,
                       flow01, back_flow01, device=None, augment_flow_type=None):
    _, h, w = img0.shape
    h_len = int(utils.get_random(h / 4, h / 4, False)) # 1/4 - 1/2
    w_len = int(utils.get_random(w / 4, w / 4, False)) # 1/4 - 1/2
    h_start = int(utils.get_random(h - h_len, 0, False))
    w_start = int(utils.get_random(w - w_len, 0, False))
    h_end = h_start + h_len
    w_end = w_start + w_len

    if augment_flow_type is None:
        augment_flow_type = utils.get_random(8, 0, False)
    fw = FW(device).to(device)
    cf = ConcatFlow(device).to(device)
    bf = BackFlow(device).to(device)
    sf = SpecialFlow(device).to(device)
    if augment_flow_type >= 5.:
        _, h, w = img0.shape
        special_flow_small, back_special_flow_small = sf((h_len, w_len), augment_flow_type)
        special_flow = torch.zeros_like(flow01).to(device)
        back_special_flow = torch.zeros_like(flow01).to(device)
        special_flow[:, h_start:h_end, w_start:w_end] = special_flow_small
        back_special_flow[:, h_start:h_end, w_start:w_end] = back_special_flow_small

        augment_img0_flow = cf(back_special_flow, special_flow, flow01, img0_depth)
        augment_img1_flow = cf(flow01, back_flow01, special_flow, img1_depth)

        img0_all = torch.cat((img0, img0_depth), axis=0)
        augment_img0_all, valid, collision = fw(img0_all, special_flow, img0_depth)
        augment_img0 = augment_img0_all[0:3]
        augment_img0 = utils.inpaint(augment_img0, valid, collision)
        augment_img0_depth = augment_img0_all[3:4]
        augment_img0_depth = utils.fix_warped_depth(augment_img0_depth)
        img1_all = torch.cat((img1, img1_depth), axis=0)
        augment_img1_all, valid, collision = fw(img1_all, special_flow, img1_depth)
        augment_img1 = augment_img1_all[0:3]
        augment_img1 = utils.inpaint(augment_img1, valid, collision)
        augment_img1_depth = augment_img1_all[3:4]
        augment_img1_depth = utils.fix_warped_depth(augment_img1_depth)

        back_augment_img0_flow = bf(augment_img0_flow, augment_img0_depth)
        back_augment_img1_flow = bf(augment_img1_flow, img0_depth)
        
        del fw, cf, bf, sf
        del special_flow, back_special_flow

        return (augment_img0, augment_img0_depth,
                augment_img0_flow, back_augment_img0_flow, 
                img1, img1_depth), (
                img0, img0_depth,
                augment_img1_flow, back_augment_img1_flow,
                augment_img1, augment_img1_depth), int(augment_flow_type)
    elif augment_flow_type >= 3.:
        pass
    elif augment_flow_type >= 0.:
        if augment_flow_type >= 2.:
            gray = torch.tensor([[0.2989, 0.2989, 0.2989],
                                 [0.5870, 0.5870, 0.5870],
                                 [0.1140, 0.1140, 0.1140]]).type(torch.float32).to(device)
            augment_img_func = lambda img, g=gray: (img.permute(1, 2, 0) @ g).permute(2, 0, 1)
        elif augment_flow_type >= 1.:
            channel = int(utils.get_random(3, 0, False))
            shift = torch.zeros_like(img0[:, h_start:h_end, w_start:w_end])
            shift[channel, :, :] = utils.get_random(10, 15)
            augment_img_func = lambda img, c=channel, s=shift: img + s 
        elif augment_flow_type >= 0.:
            scale = utils.get_random(1, 0, False)
            augment_img_func = lambda img, s=scale: img * s

        # print(f"{img0.dtype = }")
        # print(f"{img1.dtype = }")
        augment_img0 = copy.deepcopy(img0)
        augment_img1 = copy.deepcopy(img1)
        augment_img0_small = augment_img_func(img0[:, h_start:h_end, w_start:w_end])
        augment_img1_small = augment_img_func(img1[:, h_start:h_end, w_start:w_end])
        augment_img0[:, h_start:h_end, w_start:w_end] = augment_img0_small
        augment_img1[:, h_start:h_end, w_start:w_end] = augment_img1_small

        if augment_flow_type >= 2.:
            del gray
        elif augment_flow_type >= 1.:
            del channel, shift
        elif augment_flow_type >= 0.:
            del scale
        del fw, cf, bf
        del augment_img_func
            
        return (augment_img0, img0_depth,
                flow01, back_flow01, 
                img1, img1_depth), (
                img0, img0_depth,
                flow01, back_flow01, 
                augment_img1, img1_depth), int(augment_flow_type)


def augment_flow(img0, img0_depth, img1, img1_depth,
                 flow01, back_flow01, device=None, augment_flow_type=None):
    _, h, w = img0.shape
    if augment_flow_type is None:
        augment_flow_type = utils.get_random(8, 0, False)
    fw = FW(device).to(device)
    cf = ConcatFlow(device).to(device)
    bf = BackFlow(device).to(device)
    sf = SpecialFlow(device).to(device)
    if augment_flow_type >= 5.:
        _, h, w = img0.shape
        special_flow, back_special_flow = sf((h, w), augment_flow_type)
        special_flow = special_flow.to(device)
        back_special_flow = back_special_flow.to(device)
        # special_flow = special_flow.repeat(b, 1, 1, 1)
        # back_special_flow = back_special_flow.repeat(b, 1, 1, 1)

        # augment_img0_flow = cf(special_flow, back_special_flow, flow01, img0_depth)
        augment_img0_flow = cf(back_special_flow, special_flow, flow01, img0_depth)
        augment_img1_flow = cf(flow01, back_flow01, special_flow, img1_depth)

        img0_all = torch.cat((img0, img0_depth), axis=0)
        augment_img0_all, valid, collision = fw(img0_all, special_flow, img0_depth)
        augment_img0 = augment_img0_all[0:3]
        augment_img0 = utils.inpaint(augment_img0, valid, collision)
        augment_img0_depth = augment_img0_all[3:4]
        augment_img0_depth = utils.fix_warped_depth(augment_img0_depth)
        img1_all = torch.cat((img1, img1_depth), axis=0)
        augment_img1_all, valid, collision = fw(img1_all, special_flow, img1_depth)
        augment_img1 = augment_img1_all[0:3]
        augment_img1 = utils.inpaint(augment_img1, valid, collision)
        augment_img1_depth = augment_img1_all[3:4]
        augment_img1_depth = utils.fix_warped_depth(augment_img1_depth)

        back_augment_img0_flow = bf(augment_img0_flow, augment_img0_depth)
        back_augment_img1_flow = bf(augment_img1_flow, img0_depth)
        
        del fw, cf, bf, sf
        del special_flow, back_special_flow

        return (augment_img0, augment_img0_depth,
                augment_img0_flow, back_augment_img0_flow, 
                img1, img1_depth), (
                img0, img0_depth,
                augment_img1_flow, back_augment_img1_flow,
                augment_img1, augment_img1_depth), int(augment_flow_type)
    elif augment_flow_type >= 3.:
        pass
        # if augment_flow_type >= 4.:
        #     h_len = int(utils.get_random(h / 4, h / 2, False))
        #     w_len = int(utils.get_random(w / 4, w / 2, False))
        # elif augment_flow_type >= 3.:
        #     h_len = int(utils.get_random(h / 4, h / 4, False))
        #     w_len = int(utils.get_random(w / 4, w / 4, False))
        # h_start = int(utils.get_random(h - h_len, 0, False))
        # w_start = int(utils.get_random(w - w_len, 0, False))

        # mask_all = torch.ones((b, 3 + 1 + 2, h, w)).to(device)
        # mask_all[:, :, h_start:h_start + h_len, w_start:w_start + w_len] = 0

        # if augment_flow_type >= 4.:
        #     mask_all = mask_all * -1 + 1 # xor for float
        # img_mask, depth_mask, flow_mask = mask_all[:, 0:3], mask_all[:, 3:4], mask_all[:, 4:6]
        # mask_all_fw, valid, collision = fw(mask_all, flow01, img0_depth)
        # mask_all_bf, valid, collision = fw(mask_all, back_flow01, img1_depth)
        # img_mask_fw, depth_mask_fw, flow_mask_fw = mask_all_fw[:, 0:3], mask_all_fw[:, 3:4], mask_all_fw[:, 4:6]
        # img_mask_bf, depth_mask_bf, flow_mask_bf = mask_all_bf[:, 0:3], mask_all_bf[:, 3:4], mask_all_bf[:, 4:6]

        # augment_img0 = img0 * img_mask
        # augment_img0_depth = img0_depth * depth_mask
        # augment_img0_depth = utils.fix_depth(augment_img0_depth, augment_img0)
        # augment_img1 = img1 * img_mask
        # augment_img1_depth = img1_depth * depth_mask
        # augment_img1_depth = utils.fix_depth(augment_img1_depth, augment_img1)

        # augment_img0_flow = flow01 * flow_mask
        # augment_img1_flow = flow01 * flow_mask_bf
        # back_augment_img0_flow = back_flow01 * flow_mask_fw
        # back_augment_img1_flow = back_flow01 * flow_mask

        # adjust = utils.get_random(0, 1) > 0
        # adjust_img0, adjust_img0_depth = img0, img0_depth
        # adjust_img1, adjust_img1_depth = img1, img1_depth
        # if adjust:
        #     adjust_img0 = img0 * img_mask_bf
        #     adjust_img0_depth = img0_depth * depth_mask_bf
        #     adjust_img0_depth = utils.fix_depth(adjust_img0_depth, adjust_img0)
        #     adjust_img1 = img1 * img_mask_fw
        #     adjust_img1_depth = img1_depth * depth_mask_fw
        #     adjust_img1_depth = utils.fix_depth(adjust_img1_depth, adjust_img1)
        
        # del fw, cf, bf
        # del h_len, w_len, h_start, w_start
        # del img_mask, depth_mask, flow_mask
        # del mask_all, mask_all_fw, mask_all_bf
        # del img_mask_fw, depth_mask_fw, flow_mask_fw
        # del img_mask_bf, depth_mask_bf, flow_mask_bf

        # return (augment_img0, augment_img0_depth,
        #         augment_img0_flow, back_augment_img0_flow, 
        #         adjust_img1, adjust_img1_depth), (
        #         adjust_img0, adjust_img0_depth,
        #         augment_img1_flow, back_augment_img1_flow,
        #         augment_img1, augment_img1_depth), int(augment_flow_type)
    elif augment_flow_type >= 0.:
        if augment_flow_type >= 2.:
            gray = torch.tensor([[0.2989, 0.2989, 0.2989],
                                 [0.5870, 0.5870, 0.5870],
                                 [0.1140, 0.1140, 0.1140]]).type(torch.float32).to(device)
            augment_img_func = lambda img, g=gray: (img.permute(1, 2, 0) @ g).permute(2, 0, 1)
        elif augment_flow_type >= 1.:
            channel = int(utils.get_random(3, 0, False))
            shift = torch.zeros_like(img0)
            shift[channel, :, :] = utils.get_random(10, 15)
            augment_img_func = lambda img, c=channel, s=shift: img + s
        elif augment_flow_type >= 0.:
            scale = utils.get_random(1, 0, False)
            augment_img_func = lambda img, s=scale: img * s

        # print(f"{img0.dtype = }")
        # print(f"{img1.dtype = }")

        augment_img0 = augment_img_func(img0)
        augment_img1 = augment_img_func(img1)

        if augment_flow_type >= 2.:
            del gray
        elif augment_flow_type >= 1.:
            del channel, shift
        elif augment_flow_type >= 0.:
            del scale
        del fw, cf, bf
        del augment_img_func
            
        return (augment_img0, img0_depth,
                flow01, back_flow01, 
                img1, img1_depth), (
                img0, img0_depth,
                flow01, back_flow01, 
                augment_img1, img1_depth), int(augment_flow_type)

class Plausible():
    @staticmethod
    def f():
        return 1

    @staticmethod
    def B():
        return 50

    @staticmethod
    def K(size, another=False):
        h, w = size
        K = torch.tensor([[[0.58,    0, 0.5, 0],
                           # [0,    1.92, 0.5, 0],
                           [0,    0.58, 0.5, 0],
                           [0,       0,   1, 0], 
                           [0,       0,   0, 1]]], dtype=torch.float32)

        if another:
            K[:, :2, :2] *= 2

        K[:, 0, :] *= w
        K[:, 1, :] *= h

        del h, w
        return K, torch.linalg.inv(K)

    @staticmethod
    def random_motion(axisangle_range, axisangle_base, translation_range, translation_base,
                      another_axisangle=None, another_translation=None):
        ax = utils.get_random(math.pi * axisangle_range, math.pi * axisangle_base)
        ay = utils.get_random(math.pi * axisangle_range, math.pi * axisangle_base)
        az = utils.get_random(math.pi * axisangle_range, math.pi * axisangle_base)
        camera_ang = [ax, ay, az]

        cx = utils.get_random(translation_range, translation_base)
        cy = utils.get_random(translation_range, translation_base)
        cz = utils.get_random(translation_range, translation_base)
        camera_mot = [cx, cy, cz]

        axisangle = torch.tensor([[camera_ang]], dtype=torch.float32)
        translation = torch.tensor([[camera_mot]], dtype=torch.float32)

        if another_axisangle is not None and another_translation is not None:
            T = geometry.transformation_from_parameters(axisangle + another_axisangle,
                                                        translation + another_translation)
        else:
            T = geometry.transformation_from_parameters(axisangle, translation)

        del ax, ay, az
        del camera_ang, camera_mot
        return T, axisangle, translation


class Convert():
    @staticmethod
    def depth_to_disparity(depth):
        # _, _, w = depth.shape
        s = utils.get_random(0.3, 0.8, random_sign=False)
        B, f = Plausible.B(), Plausible.f()

        disparity = s * B * f / depth
        del B, f
        del s
        return disparity

    @staticmethod
    def disparity_to_flow(disparity, random_sign=True):
        flow = torch.cat((disparity, torch.zeros_like(disparity)), axis=0) * -1
        if random_sign:
            flow = flow * utils.get_random(0, 1)
        return flow

    @staticmethod
    def disparity_to_depth(disparity):
        B, f = Plausible.B(), Plausible.f()
        depth = B * f / (disparity + 0.005)

        del B, f
        return depth

    @staticmethod
    def depth_to_random_flow(depth, device=None, segment=None):
        _, h, w = depth.shape
        depth = depth.unsqueeze(0)
        K, inv_K = Plausible.K((h, w))
        K, inv_K = K.to(device), inv_K.to(device)
        backproject_depth = geometry.BackprojectDepth(1, h, w, device=device)
        project_3d = geometry.Project3D(1, h, w)

        with torch.no_grad():
            cam_points = backproject_depth(depth, inv_K)

        T1, axisangle, translation = Plausible.random_motion(1. / 36., 1. / 36., 0.1, 0.1) 
        T1 = T1.to(device)

        with torch.no_grad():
            p1, z1 = project_3d(cam_points, K, T1)
        z1 = z1.reshape(1, h, w)
            
        if segment is not None:
            instances, labels = segment
            for l in range(len(labels)):
                Ti, _, _ = Plausible.random_motion(1. / 72., 1. / 72., 0.05, 0.05,
                                                    axisangle, translation) 
                pi, zi = project_3d(cam_points, K, Ti)
                zi = zi.reshape(1, h, w)
                p1[instances[l] > 0] = pi[instances[l] > 0]
                z1[instances[l][:, :, :, 0] > 0] = zi[instances[l][:, :, :, 0] > 0]

        p1 = (p1 + 1) / 2
        p1[:, :, :, 0] *= w - 1
        p1[:, :, :, 1] *= h - 1
        # p1 = p1.double()

        meshgrid = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        p0 = torch.stack(meshgrid, axis=-1).type(torch.float32).to(device)
        flow = p1 - p0
        flow = flow.permute(0, 3, 1, 2)

        del K, inv_K
        del T1, p1, z1, axisangle, translation
        if segment is not None:
            del Ti, pi, zi
        del meshgrid, p0
        del backproject_depth, project_3d

        return flow.squeeze(0)


class ConcatFlow(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.fw = FW(device).to(device)
    
    def forward(self, flowAB, back_flowAB, flowBC, imgB_depth):
        with torch.no_grad():
            concat_flow, valid, collision = self.fw(flowBC, back_flowAB, imgB_depth)
        concat_flow = concat_flow + flowAB
        return concat_flow.to(self.device)

class BackFlow(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.fw = FW(device).to(device)

    def forward(self, flowAB, imgA_depth):
        with torch.no_grad():
            back_flow, valid, collision = self.fw(flowAB, flowAB, imgA_depth)
        back_flow = back_flow * -1.
        return back_flow.to(self.device)


class PreprocessPlusAugment(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.fw = FW(self.device).to(self.device)
        self.cf = ConcatFlow(self.device).to(self.device)
        self.bf = BackFlow(self.device).to(self.device)       
        self.fw.eval()
        self.cf.eval()
        self.bf.eval()

    # def forward(self, img0, img0_depth, output_dir):
    def forward(self, datas, output_dir, is_stereo=False, augment_small=False):
        start_preprocess_time = time.time()
        print(f"{output_dir = }")

        if not is_stereo:
            img0, img0_depth = datas
            img0 = img0.to(self.device)
            img0_depth = img0_depth.to(self.device)
            # img0_depth = utils.fix_warped_depth(img0_depth)
            img0_depth = utils.normalize_depth(img0_depth)
            disp0 = Convert.depth_to_disparity(img0_depth)
            flow01 = Convert.disparity_to_flow(disp0).to(self.device)
            img0_all = torch.cat((img0, img0_depth, flow01 * -1), axis=0) 
            img0_all_fw, valid, collision = self.fw(img0_all, flow01, img0_depth)
            img0_all_fw = img0_all_fw.squeeze(0) 
            img1, img1_depth, back_flow01 = img0_all_fw[0:3], img0_all_fw[3:4], img0_all_fw[4:6]
            img1 = utils.inpaint(img1, valid, collision)
            img1_depth = utils.fix_warped_depth(img1_depth)
        else:
            img0, img1, disp0, disp1 = datas
            img0 = img0.to(self.device)
            img1 = img1.to(self.device)
            img1_real = img1
            img0_depth = Convert.disparity_to_depth(disp0).to(device)
            # img0_depth = utils.fix_warped_depth(img0_depth)
            img0_depth = utils.normalize_depth(img0_depth)
            flow01 = Convert.disparity_to_flow(disp0, random_sign=False).to(device)
            has_disp1 = (disp1 is not None)
            if has_disp1:
                img1_depth = Convert.disparity_to_depth(disp1).to(device)
                # img1_depth = utils.fix_warped_depth(img1_depth)
                img1_depth = utils.normalize_depth(img1_depth)
                back_flow01 = Convert.disparity_to_flow(disp1, random_sign=True).to(device) * -1
            else:
                img0_all = torch.cat((img0, img0_depth, flow01 * -1), axis=0) 
                img0_all_fw, valid, collision = self.fw(img0_all, flow01, img0_depth)
                img0_all_fw = img0_all_fw.squeeze(0)
                img1, img1_depth, back_flow01 = img0_all_fw[0:3], img0_all_fw[3:4], img0_all_fw[4:6]
                img1_depth = utils.fix_warped_depth(img1_depth)
                img1 = utils.inpaint(img1, valid, collision)

        # unify mono: (img0, img0_depth), and stero: (img0, img0, disp0, disp1)
        # to (img0, img0_depth, flow01, back_flow01, img1, img1_depth)

        _, h, w = img0.shape
        
        flow12 = Convert.depth_to_random_flow(img1_depth, self.device)
        img1_all = torch.cat((img1, img1_depth, flow12 * -1), axis=0) 
        img1_all_fw, valid, collision = self.fw(img1_all, flow12, img1_depth)
        img1_all_fw = img1_all_fw.squeeze(0) 
        img2, img2_depth, back_flow12 = img1_all_fw[0:3], img1_all_fw[3:4], img1_all_fw[4:6]
        img2 = utils.inpaint(img2, valid, collision)
        img2_depth = utils.fix_warped_depth(img2_depth)

        flow02 = self.cf(flow01, back_flow01, flow12, img1_depth)
        back_flow02 = self.bf(flow02, img0_depth)
        concat_img, valid, collision = self.fw(img0, flow02, img0_depth)
        concat_img = utils.inpaint(concat_img, valid, collision)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        to_save_data0 = torch.cat((img0, img0_depth, img1, img1_depth, img2, img2_depth,
                                   flow01, back_flow01, flow12, back_flow12,
                                   flow02, back_flow02), axis=0).cpu().numpy()
                                 #  0:3,        3:4,  4:7,        7:8, 8:11,      11:12,
                                 #  12:14,       14:16,  16:18,       18:20,
                                 #  20:22,       22:24

        if to_save_data0.shape != (24, h, w):
            print(f"something wrong when preprocess {output_dir}")
            sys.exit()

        np.savez_compressed(f"{output_dir}/group.npz",
                            img_depth_flow=to_save_data0)

        end_preprocess_time = time.time()
        print(f"preprocess time = {end_preprocess_time - start_preprocess_time}")

        start_augment_time = time.time()
        group = [(img0, img0_depth, img1, img1_depth, flow01, back_flow01),
                 (img1, img1_depth, img2, img2_depth, flow12, back_flow12),
                 (img0, img0_depth, img2, img2_depth, flow02, back_flow02)]
        
        # group_output_dir = f"test_output/group"
        # if not os.path.exists(group_output_dir):
        #     os.makedirs(group_output_dir)

        # cv2.imwrite(f"{group_output_dir}/img0.png", img0.permute(1, 2, 0).cpu().numpy())
        # plt.imsave(f"{group_output_dir}/img0_depth.png", 1 / img0_depth[0].cpu().numpy(), cmap="magma")
        # cv2.imwrite(f"{group_output_dir}/img1.png", img1.permute(1, 2, 0).cpu().numpy())
        # # cv2.imwrite(f"{group_output_dir}/img1_real.png", img1_real.permute(1, 2, 0).cpu().numpy())
        # plt.imsave(f"{group_output_dir}/img1_depth.png", 1 / img1_depth[0].cpu().numpy(), cmap="magma")
        # cv2.imwrite(f"{group_output_dir}/img2.png", img2.permute(1, 2, 0).cpu().numpy())
        # plt.imsave(f"{group_output_dir}/img2_depth.png", 1 / img2_depth[0].cpu().numpy(), cmap="magma")
        # cv2.imwrite(f"{group_output_dir}/flow01.png", utils.color_flow(flow01.permute(1, 2, 0).unsqueeze(0).cpu())[1])
        # cv2.imwrite(f"{group_output_dir}/flow12.png", utils.color_flow(flow12.permute(1, 2, 0).unsqueeze(0).cpu())[1])
        # cv2.imwrite(f"{group_output_dir}/flow02.png", utils.color_flow(flow02.permute(1, 2, 0).unsqueeze(0).cpu())[1])

        # img0_mask = (img0_depth != 100).repeat(3, 1, 1)
        # img1_mask = (img1_depth != 100).repeat(3, 1, 1)
        # img2_mask = (img2_depth != 100).repeat(3, 1, 1)
        # cv2.imwrite(f"{group_output_dir}/img0_mask.png", img0_mask.permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(f"{group_output_dir}/img1_mask.png", img1_mask.permute(1, 2, 0).cpu().numpy() * 255)
        # cv2.imwrite(f"{group_output_dir}/img2_mask.png", img2_mask.permute(1, 2, 0).cpu().numpy() * 255)

        # cv2.imwrite(f"{group_output_dir}/img0_masked.png", (img0 * img0_mask).permute(1, 2, 0).cpu().numpy())
        # cv2.imwrite(f"{group_output_dir}/img1_masked.png", (img1 * img1_mask).permute(1, 2, 0).cpu().numpy())
        # cv2.imwrite(f"{group_output_dir}/img2_masked.png", (img2 * img2_mask).permute(1, 2, 0).cpu().numpy())

        # warped_img1, _, _ = self.fw(img0, flow01, img0_depth)
        # warped_img2, _, _ = self.fw(img0, flow02, img0_depth)
        # cv2.imwrite(f"{group_output_dir}/warped_img1.png", warped_img1.permute(1, 2, 0).cpu().numpy())
        # cv2.imwrite(f"{group_output_dir}/warped_img2.png", warped_img2.permute(1, 2, 0).cpu().numpy())

        for group_idx, (imgA, imgA_depth, imgB, imgB_depth, flowAB, back_flowAB) in enumerate(group):
            for augment_idx, augment_flow_type in enumerate([0, 5, 6, 7, 1, 5, 6, 7, 2, 5, 6, 7]):
            # for augment_idx, augment_flow_type in enumerate([0, 1, 2, 5, 6, 7]):
            # for augment_idx, augment_flow_type in enumerate([5, 6, 7]):

                if not augment_small:
                    set1, set2, _ = augment_flow(imgA, imgA_depth, imgB, imgB_depth, flowAB,
                                                 back_flowAB, device=self.device,
                                                 augment_flow_type=augment_flow_type)
                else:
                    set1, set2, _ = augment_flow_small(imgA, imgA_depth, imgB, imgB_depth, flowAB,
                                                 back_flowAB, device=self.device,
                                                 augment_flow_type=augment_flow_type)

                to_save_data1 = torch.cat([set1[i] for i in range(0, 4)], axis=0) # img0, img0_depth, flow01, back_flow01
                                                                                  #  0:3         3:4     4:6          6:8
                to_save_data2 = torch.cat([set2[i] for i in range(2, 6)], axis=0) # flow01, back_flow01, img1, img1_depth
                                                                                  #    0:2          2:4   4:7         7:8

                to_save_data1_np = to_save_data1.detach().cpu().numpy()
                to_save_data2_np = to_save_data2.detach().cpu().numpy()

                if to_save_data1_np.shape != (8, h, w) or to_save_data2_np.shape != (8, h, w):
                    print(f"something wrong when preprocess {output_dir}")
                    sys.exit()

                np.savez_compressed(f"{output_dir}/{group_idx}_{augment_idx}_1.npz",
                                    augment_img=0,
                                    img_depth_flow=to_save_data1_np,
                                    augment_flow_type=augment_flow_type)
                np.savez_compressed(f"{output_dir}/{group_idx}_{augment_idx}_2.npz",
                                    augment_img=1,
                                    img_depth_flow=to_save_data2_np,
                                    augment_flow_type=augment_flow_type)
                
                # set1_output_dir = f"test_output/{group_idx}_{augment_idx}_1"
                # set2_output_dir = f"test_output/{group_idx}_{augment_idx}_2"
                # if not os.path.exists(set1_output_dir):
                #     os.makedirs(set1_output_dir)
                # if not os.path.exists(set2_output_dir):
                #     os.makedirs(set2_output_dir)

                # set1_warped_img, _, _ = self.fw(set1[0], set1[2], set1[1])
                # set2_warped_img, _, _ = self.fw(set2[0], set2[2], set2[1])

                # cv2.imwrite(f"{set1_output_dir}/img0.png", set1[0].permute(1, 2, 0).cpu().numpy())
                # plt.imsave(f"{set1_output_dir}/img0_depth.png", 1 / set1[1][0].cpu().numpy(), cmap="magma")
                # cv2.imwrite(f"{set1_output_dir}/img1.png", set1[4].permute(1, 2, 0).cpu().numpy())
                # plt.imsave(f"{set1_output_dir}/img1_depth.png", 1 / set1[5][0].cpu().numpy(), cmap="magma")
                # cv2.imwrite(f"{set1_output_dir}/warped_img1.png", set1_warped_img.permute(1, 2, 0).cpu().numpy())
                # cv2.imwrite(f"{set2_output_dir}/img0.png", set2[0].permute(1, 2, 0).cpu().numpy())
                # plt.imsave(f"{set2_output_dir}/img0_depth.png", 1 / set2[1][0].cpu().numpy(), cmap="magma")
                # cv2.imwrite(f"{set2_output_dir}/img1.png", set2[4].permute(1, 2, 0).cpu().numpy())
                # plt.imsave(f"{set2_output_dir}/img1_depth.png", 1 / set2[5][0].cpu().numpy(), cmap="magma")
                # cv2.imwrite(f"{set2_output_dir}/warped_img1.png", set2_warped_img.permute(1, 2, 0).cpu().numpy())
                # cv2.imwrite(f"{set1_output_dir}/flow01.png", utils.color_flow(set1[2].permute(1, 2, 0).unsqueeze(0).cpu())[1])
                # cv2.imwrite(f"{set2_output_dir}/flow01.png", utils.color_flow(set2[2].permute(1, 2, 0).unsqueeze(0).cpu())[1])

        end_augment_time = time.time()
        print(f"augmenting time = {end_augment_time - start_augment_time}")

        del augment_idx, augment_flow_type
        del imgA, imgB, imgA_depth, imgB_depth
        del flowAB, back_flowAB
        del set1, set2
        del to_save_data1, to_save_data2
        del to_save_data1_np, to_save_data2_np

        del img0, img1, img2, concat_img
        del img0_depth, img1_depth, img2_depth
        del disp0

        if is_stereo:
            del disp1
        else:
            del img0_all, img0_all_fw

        del img1_all, img1_all_fw
        del flow01, flow12, flow02
        del back_flow01, back_flow12, back_flow02
        # del fw, cf, bf
        del output_dir

        del start_augment_time, end_augment_time
        del start_preprocess_time, end_preprocess_time


def read_args():
    parser = ArgumentParser()

    parser.add_argument('--dataset')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--split', default=1, type=int)
    parser.add_argument('--split_id', default=0, type=int)
    parser.add_argument('--range', default=2, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()
    seed_mapping = {"ReDWeb": 0, "ETH3D": 10, "DIML": 30, "test_ReDWeb": 20, "filted_ReDWeb": 40}
    # utils.set_seed(12345 + args.split_id + seed_mapping[args.dataset])
    print(args)

    if args.dataset == "DIML":
        dataset = DIML("datasets/DIML")
        output_dir = "AugmentedDatasets/DIML"
        is_stereo = True
    elif args.dataset == "filted_ReDWeb":
        dataset = FiltedReDWeb("datasets/ReDWeb_V1")
        output_dir = "AugmentedDatasets/filted_ReDWeb"
        is_stereo = False


    device = f"cuda:{args.gpu}"

    ppa = PreprocessPlusAugment(device=device)

    split_len = len(dataset) / args.split
    start = int(args.split_id * split_len)
    end = int((args.split_id + 1) * split_len)
    if args.split_id == args.split - 1:
        end = len(dataset)

    print(f"{len(dataset) = }, {is_stereo = }, {start = }, {end = }")

    resize = T.Resize((480, 640))
    for epoch_idx, augment_small in enumerate([False, True]):
        if args.range == 0 and epoch_idx == 1:
            break
        if args.range == 1 and epoch_idx == 0:
            continue

        for img_idx in range(start, end):
            print(f"{img_idx + 1} / {len(dataset)}: {augment_small = }")
            utils.set_seed(12345 + img_idx + epoch_idx * len(dataset))
            datas = dataset[img_idx]

            if args.dataset == "filted_ReDWeb":
                img0, img0_depth = datas
                img0 = resize(img0)
                img0_depth = resize(img0_depth)
                datas = (img0, img0_depth)

            # output_dir = output_dirs[int(img_idx / EVERY_IMAGES_CHANGE_OUTPUT_DIR)]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            ppa(datas, f"{output_dir}/{img_idx + epoch_idx * len(dataset)}", is_stereo, augment_small)
            # break
        # break
    
    
    # areas = np.load("areas.npy")
    # hs = np.load("hs.npy")
    # ws = np.load("ws.npy")
    # print(f"{len(areas) = }")
    # print(f"{np.mean(areas) = }, {np.var(areas) = }, {np.min(areas) = }, {np.max(areas) = }")
    # print(f"{np.mean(hs) = }, {np.var(hs) = }, {np.min(hs) = }, {np.max(hs) = }")
    # print(f"{np.mean(ws) = }, {np.var(ws) = }, {np.min(ws) = }, {np.max(ws) = }")

    # h_limit = 350
    # w_limit = 350

    # # condition = (areas >= h_limit * w_limit)
    # condition = ((hs >= h_limit) & (ws >= w_limit)) 

    # areas_a = areas[condition]
    # hs_a = hs[condition]
    # ws_a = ws[condition]
    # print(f"{len(areas_a) = }")
    # print(f"{np.mean(areas_a) = }, {np.var(areas_a) = }, {np.min(areas_a) = }, {np.max(areas_a) = }")
    # print(f"{np.mean(hs_a) = }, {np.var(hs_a) = }, {np.min(hs_a) = }, {np.max(hs_a) = }")
    # print(f"{np.mean(ws_a) = }, {np.var(ws_a) = }, {np.min(ws_a) = }, {np.max(ws_a) = }")

    # print(f"{np.sum(areas_a >= 480 * 640) = }")
    # print(f"{np.sum(areas_a >= 384 * 512) = }")
    # print(f"{np.sum(areas_a >= 368 * 496) = }")

    # np.save("areas.npy", areas)
    # np.save("hs.npy", hs)
    # np.save("ws.npy", ws)
