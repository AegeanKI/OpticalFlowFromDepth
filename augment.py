import utils
import preprocess
import numpy as np
import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision.transforms.functional as F
# from torchvision.models.optical_flow import Raft_Large_Weights
# from torchvision.models.optical_flow import raft_large
import cv2
import matplotlib.pyplot as plt
import pykitti
# from parser import DataParser

from preprocess import Convert, ConcatFlow, BackFlow
from bilateral_filter import sparse_bilateral_filtering
# from using_clib import Resample2d2
# from using_clib import ForwardWarping
import glob
import os

from my_cuda.fw import FW

class SpecialFlow(nn.Module):
    def __init__(self, size, device="cuda"):
        super().__init__()
        self.h, self.w = size
        self.device = device

        meshgrid = torch.meshgrid(torch.arange(self.w), torch.arange(self.h), indexing="xy")
        self.p0 = torch.stack(meshgrid, axis=-1).type(torch.float32).to(device)
        # meshgrid = np.meshgrid(range(self.w), range(self.h), indexing="xy")
        # self.p0 = np.stack(meshgrid, axis=-1).astype(np.float32)

    def forward(self, augment_flow_type):
        if augment_flow_type >= 7.:
            p1, p_prev = self._shear()
        elif augment_flow_type >= 6.:
            p1, p_prev = self._rotate()
        elif augment_flow_type >= 5.:
            p1, p_prev = self._flip()

        special_flow = (p1 - self.p0).permute(2, 0, 1).unsqueeze(0)
        # back_special_flow = (self.p0 - p_prev).permute(2, 0, 1).unsqueeze(0)
        back_special_flow = (p_prev - self.p0).permute(2, 0, 1).unsqueeze(0)

        del p1, p_prev
        return special_flow, back_special_flow

    def _flip(self):
        horizontal = utils.get_random(0, 1) > 0

        if horizontal:
            meshgrid = torch.meshgrid(torch.arange(self.w - 1, -1, -1), torch.arange(self.h), indexing="xy")
        else:
            meshgrid = torch.meshgrid(torch.arange(self.w), torch.arange(self.h - 1, -1, -1), indexing="xy")

        p1 = torch.stack(meshgrid, axis=-1).type(torch.float32).to(self.device)
        p_prev = p1

        del horizontal, meshgrid
        return p1, p_prev

    def _rotate(self):
        c0 = (utils.get_random(self.w / 4, self.w / 2) + self.w / 2, utils.get_random(self.h / 4, self.h / 2) + self.h / 2)
        c0 = torch.tensor(c0).to(self.device)
        theta = torch.deg2rad(utils.get_random(2, 8)).to(self.device)

        rotate = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]]).type(torch.float32).to(self.device)
        reverse_rotate = torch.tensor([[torch.cos(-theta), -torch.sin(-theta)],
                                       [torch.sin(-theta), torch.cos(-theta)]]).type(torch.float32).to(self.device)

        p1 = (self.p0 - c0) @ rotate + c0
        p_prev = (self.p0 - c0) @ reverse_rotate + c0

        del c0, theta
        del rotate, reverse_rotate
        return p1, p_prev

    def _shear(self):
        horizontal = utils.get_random(0, 1) > 0
        shear_range = utils.get_random(0.15, 0.2)

        if horizontal:
            shear = torch.tensor([[1, 0], [shear_range, 1]]).type(torch.float32).to(self.device)
            reverse_shear = torch.tensor([[1, 0], [-shear_range, 1]]).type(torch.float32).to(self.device)
        else:
            shear = torch.tensor([[1, shear_range], [0, 1]]).type(torch.float32).to(self.device)
            reverse_shear = torch.tensor([[1, -shear_range], [0, 1]]).type(torch.float32).to(self.device)
        
        p1 = self.p0 @ shear
        p_prev = self.p0 @ reverse_shear

        del horizontal, shear_range
        del shear, reverse_shear
        return p1, p_prev

# class AugmentFlow(nn.Module):
#     def __init__(self, size, batch_size, device="cuda"):
#         super().__init__()
#         self.size = size
#         self.h, self.w = size
#         self.batch_size = batch_size
#         self.sf = SpecialFlow(size).to(device)

#     def forward(self, img0, img0_depth, img1, img1_depth, flow01, back_flow01):
#         augment_flow_type = utils.get_random(8, 0, False)
#         augment_flow_type = 5.2
#         if augment_flow_type >= 5.:
#             special_flow, back_special_flow = self.sf(augment_flow_type)
#             special_flow = special_flow.repeat(self.batch_size, 1, 1, 1)



def augment_flow(img0, img0_depth, img1, img1_depth,
                 flow01, back_flow01, device="cuda", augment_flow_type=None):
    b, _, h, w = img0.shape
    size = (h, w)
    if augment_flow_type is None:
        augment_flow_type = utils.get_random(8, 0, False)
    fw = FW(size, b, device).to(device)
    cf = ConcatFlow(size, b, device).to(device)
    bf = BackFlow(size, b, device).to(device)
    sf = SpecialFlow(size, device).to(device)
    if augment_flow_type >= 5.:
        special_flow, back_special_flow = sf(augment_flow_type)
        special_flow = special_flow.to(device)
        back_special_flow = back_special_flow.to(device)
        special_flow = special_flow.repeat(b, 1, 1, 1)
        back_special_flow = back_special_flow.repeat(b, 1, 1, 1)

        # augment_img0_flow = cf(special_flow, back_special_flow, flow01, img0_depth)
        augment_img0_flow = cf(back_special_flow, special_flow, flow01, img0_depth)
        augment_img1_flow = cf(flow01, back_flow01, special_flow, img1_depth)

        augment_img0 = fw(img0, special_flow, img0_depth)
        augment_img0_depth = fw(img0_depth, special_flow, img0_depth)
        augment_img0_depth = utils.fix_depth(augment_img0_depth, augment_img0)
        augment_img1 = fw(img1, special_flow, img1_depth)
        augment_img1_depth = fw(img1_depth, special_flow, img1_depth)
        augment_img1_depth = utils.fix_depth(augment_img1_depth, augment_img1)

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
        if augment_flow_type >= 4.:
            h_len = int(utils.get_random(h / 4, h / 2, False))
            w_len = int(utils.get_random(w / 4, w / 2, False))
        elif augment_flow_type >= 3.:
            h_len = int(utils.get_random(h / 4, h / 4, False))
            w_len = int(utils.get_random(w / 4, w / 4, False))
        h_start = int(utils.get_random(h - h_len, 0, False))
        w_start = int(utils.get_random(w - w_len, 0, False))

        mask_all = torch.ones((b, 3 + 1 + 2, h, w)).to(device)
        mask_all[:, :, h_start:h_start + h_len, w_start:w_start + w_len] = 0

        if augment_flow_type >= 4.:
            mask_all = mask_all * -1 + 1 # xor for float
        img_mask, depth_mask, flow_mask = mask_all[:, 0:3], mask_all[:, 3:4], mask_all[:, 4:6]
        mask_all_fw = fw(mask_all, flow01, img0_depth)
        mask_all_bf = fw(mask_all, back_flow01, img1_depth)
        img_mask_fw, depth_mask_fw, flow_mask_fw = mask_all_fw[:, 0:3], mask_all_fw[:, 3:4], mask_all_fw[:, 4:6]
        img_mask_bf, depth_mask_bf, flow_mask_bf = mask_all_bf[:, 0:3], mask_all_bf[:, 3:4], mask_all_bf[:, 4:6]

        augment_img0 = img0 * img_mask
        augment_img0_depth = img0_depth * depth_mask
        augment_img0_depth = utils.fix_depth(augment_img0_depth, augment_img0)
        augment_img1 = img1 * img_mask
        augment_img1_depth = img1_depth * depth_mask
        augment_img1_depth = utils.fix_depth(augment_img1_depth, augment_img1)

        augment_img0_flow = flow01 * flow_mask
        augment_img1_flow = flow01 * flow_mask_bf
        back_augment_img0_flow = back_flow01 * flow_mask_fw
        back_augment_img1_flow = back_flow01 * flow_mask

        adjust = utils.get_random(0, 1) > 0
        adjust_img0, adjust_img0_depth = img0, img0_depth
        adjust_img1, adjust_img1_depth = img1, img1_depth
        if adjust:
            adjust_img0 = img0 * img_mask_bf
            adjust_img0_depth = img0_depth * depth_mask_bf
            adjust_img0_depth = utils.fix_depth(adjust_img0_depth, adjust_img0)
            adjust_img1 = img1 * img_mask_fw
            adjust_img1_depth = img1_depth * depth_mask_fw
            adjust_img1_depth = utils.fix_depth(adjust_img1_depth, adjust_img1)
        
        del fw, cf, bf
        del h_len, w_len, h_start, w_start
        del img_mask, depth_mask, flow_mask
        del mask_all, mask_all_fw, mask_all_bf
        del img_mask_fw, depth_mask_fw, flow_mask_fw
        del img_mask_bf, depth_mask_bf, flow_mask_bf

        return (augment_img0, augment_img0_depth,
                augment_img0_flow, back_augment_img0_flow, 
                adjust_img1, adjust_img1_depth), (
                adjust_img0, adjust_img0_depth,
                augment_img1_flow, back_augment_img1_flow,
                augment_img1, augment_img1_depth), int(augment_flow_type)
    elif augment_flow_type >= 0.:
        if augment_flow_type >= 2.:
            gray = torch.tensor([[0.2989, 0.2989, 0.2989],
                             [0.5870, 0.5870, 0.5870],
                             [0.1140, 0.1140, 0.1140]]).type(torch.float32).to(device)
            augment_img_func = lambda img, g=gray: (img.permute(0, 2, 3, 1) @ g).permute(0, 3, 1, 2)
        elif augment_flow_type >= 1.:
            channel = int(utils.get_random(3, 0, False))
            shift = torch.zeros_like(img0)
            shift[:, channel, :, :] = utils.get_random(10, 15)
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


# def augment(img0, img0_depth, img1, img1_depth, flow01, back_flow01, device):
#     set1, set2, augment_flow_type = augment_flow(img0, img0_depth, img1, img1_depth,
#                                                  flow01, back_flow01, device)
#     return set1, set2, augment_flow

#     (augment_img0, augment_img0_depth,
#     augment_img0_flow, back_augment_img0_flow,
#     img1, img1_depth) = set1
#     (img0, img0_depth,
#     augment_img1_flow, back_augment_img1_flow,
#     augment_img1, augment_img1_depth) = set2


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    batch_size = 4
    every_batch_change_dir = 160 // batch_size

    for batch_idx in range(20000 // batch_size):
        batch_dir = f"output/stereo/{batch_idx // every_batch_change_dir}"
        
        npz_file = np.load(f"{batch_dir}/{batch_idx}.npz")
        img_depth_flow = npz_file['img_depth_flow']
        # original_size = npz_file['original_size']
        # print(f"{img_depth_flow.shape = }")
        # print(f"{original_size.shape = }")

        # img_depth_flow = torch.from_numpy(img_depth_flow).to(device)

        # img0 = img_depth_flow[:, 0:3]
        # img1 = img_depth_flow[:, 3:6]
        # img2 = img_depth_flow[:, 6:9]
        # img0_depth = img_depth_flow[:, 9:10]
        # img1_depth = img_depth_flow[:, 10:11]
        # img2_depth = img_depth_flow[:, 11:12]
        # concat_img = img_depth_flow[:, 12:15]
        # flow01 = img_depth_flow[:, 15:17]
        # flow12 = img_depth_flow[:, 17:19]
        # flow02 = img_depth_flow[:, 19:21]
        # back_flow01 = img_depth_flow[:, 21:23]
        # back_flow12 = img_depth_flow[:, 23:25]
        # back_flow02 = img_depth_flow[:, 25:27]

        # print(f"{img0.shape = }")
        # print(f"{img0_depth.shape = }")
        # print(f"{flow01.shape = }")
        # print(f"{back_flow01.shape = }")

        set1, set2, augment_flow_type = augment_flow(img0, img0_depth, img1, img1_depth,
                                                     flow01, back_flow01, device)


    quit()


    # augment(img0, img0_depth, img1, img1_depth, flow01, back_flow01, size, f"testing/{img_name}/01", batch_size, device)
    # augment(img1, img1_depth, img2, img2_depth, flow12, back_flow12, size, f"output/augment/{img_name}/12", batch_size)
    # augment(img0, img0_depth, img2, img2_depth, flow02, back_flow02, size, f"output/augment/{img_name}/02", batch_size)










