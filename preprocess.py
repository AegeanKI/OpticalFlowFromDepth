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
from alt_cuda.fw import FW
from collections import defaultdict
import random
from argparse import ArgumentParser
from dataloader import DIML, ReDWeb, COCO
import copy

class SpecialFlow(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.horizontal_flip = True
        self.horizontal_shear = True

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
        self.horizontal_flip = not self.horizontal_flip

        if self.horizontal_flip:
            meshgrid = torch.meshgrid(torch.arange(w - 1, -1, -1), torch.arange(h), indexing="xy")
        else:
            meshgrid = torch.meshgrid(torch.arange(w), torch.arange(h - 1, -1, -1), indexing="xy")

        p1 = torch.stack(meshgrid, axis=-1).type(torch.float32).to(self.device)
        p_prev = p1

        del meshgrid
        return p1, p_prev

    def _rotate(self, size):
        h, w = size
        c0 = (utils.get_random(w / 4, w / 2) + w / 2, utils.get_random(h / 4, h / 2) + h / 2)
        c0 = torch.tensor(c0).to(self.device)
        theta = torch.deg2rad(utils.get_random(2, 8)).to(self.device)

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
        self.horizontal_shear = not self.horizontal_shear
        shear_range = utils.get_random(0.15, 0.2)

        if self.horizontal_shear:
            shear = torch.tensor([[1, 0], [shear_range, 1]]).type(torch.float32).to(self.device)
            reverse_shear = torch.tensor([[1, 0], [-shear_range, 1]]).type(torch.float32).to(self.device)
        else:
            shear = torch.tensor([[1, shear_range], [0, 1]]).type(torch.float32).to(self.device)
            reverse_shear = torch.tensor([[1, -shear_range], [0, 1]]).type(torch.float32).to(self.device)
        
        p0 = self._get_p0(size)
        p1 = p0 @ shear
        p_prev = p0 @ reverse_shear

        del shear_range
        del shear, reverse_shear
        return p1, p_prev

    def _get_p0(self, size):
        h, w = size
        meshgrid = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        p0 = torch.stack(meshgrid, axis=-1).type(torch.float32).to(self.device)
        return p0

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
        augment_img0_flow, augment_img0_flow_valid = cf(back_special_flow, special_flow, flow01, img0_depth)
        augment_img1_flow, augment_img1_flow_valid = cf(flow01, back_flow01, special_flow, img1_depth)

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

        back_augment_img0_flow, back_augment_img0_flow_valid = bf(augment_img0_flow, augment_img0_depth)
        back_augment_img1_flow, back_augment_img1_flow_valid = bf(augment_img1_flow, img0_depth)
        
        del fw, cf, bf, sf

        return (augment_img0, augment_img0_depth,
                augment_img0_flow, back_augment_img0_flow, 
                img1, img1_depth), (
                img0, img0_depth,
                augment_img1_flow, back_augment_img1_flow,
                augment_img1, augment_img1_depth), int(augment_flow_type), (special_flow, back_special_flow)
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
            shift = torch.zeros_like(img0)
            shift[channel, :, :] = utils.get_random(10, 15)
            augment_img_func = lambda img, c=channel, s=shift: img + s
        elif augment_flow_type >= 0.:
            scale = utils.get_random(1, 0, False)
            augment_img_func = lambda img, s=scale: img * s

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
                augment_img1, img1_depth), int(augment_flow_type), None

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
        s = utils.get_random(0.3, 0.8, random_sign=False)
        B, f = Plausible.B(), Plausible.f()

        disparity = s * B * f / depth
        del B, f
        del s
        return disparity

    @staticmethod
    def disparity_to_flow(disparity, device=None, random_sign=True):
        flow = torch.cat((disparity, torch.zeros_like(disparity)), axis=0) * -1.0
        if random_sign:
            flow = flow * utils.get_random(0, 1)
        flow = flow.to(device)
        return flow

    @staticmethod
    def disparity_to_depth(disparity):
        B, f = Plausible.B(), Plausible.f()
        depth = B * f / (disparity + 0.005)

        del B, f
        return depth

    @staticmethod
    def depth_to_random_flow(depth, device=None, segment=None, T1=None):
        _, h, w = depth.shape
        depth = depth.unsqueeze(0)
        K, inv_K = Plausible.K((h, w))
        K, inv_K = K.to(device), inv_K.to(device)
        backproject_depth = geometry.BackprojectDepth(1, h, w, device=device)
        project_3d = geometry.Project3D(1, h, w)

        with torch.no_grad():
            cam_points = backproject_depth(depth, inv_K)

        if T1 is None:
            T1, _, _ = Plausible.random_motion(1. / 36., 1. / 36., 0.1, 0.1) 
            T1 = T1.to(device)

        with torch.no_grad():
            p1, z1 = project_3d(cam_points, K, T1)
        z1 = z1.reshape(1, h, w)
            
        p1 = (p1 + 1) / 2
        p1[:, :, :, 0] *= w - 1
        p1[:, :, :, 1] *= h - 1

        meshgrid = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        p0 = torch.stack(meshgrid, axis=-1).type(torch.float32).to(device)
        flow = p1 - p0
        flow = flow.permute(0, 3, 1, 2)

        del K, inv_K
        del p1, z1
        del meshgrid, p0
        del backproject_depth, project_3d

        return flow.squeeze(0), T1
    

class ConcatFlow(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.fw = FW(device).to(device)
    
    def forward(self, flowAB, back_flowAB, flowBC, imgB_depth):
        with torch.no_grad():
            concat_flow, valid, collision = self.fw(flowBC, back_flowAB, imgB_depth)
            # print(f"{concat_flow.shape = }")
            # print(f"{valid.shape = }")
        concat_flow = (concat_flow + flowAB) * valid
        return concat_flow.to(self.device), valid.to(self.device)

class BackFlow(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.fw = FW(device).to(device)

    def forward(self, flowAB, imgA_depth):
        with torch.no_grad():
            back_flow, valid, collision = self.fw(flowAB, flowAB, imgA_depth)
        back_flow = (back_flow * -1.0) * valid
        # return back_flow.to(self.device)
        return back_flow.to(self.device), valid.to(self.device)


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
    def forward(self, datas, output_dir, is_stereo=False, n_continuous=4):
        n_continuous = n_continuous - 1
        start_preprocess_time = time.time()
        print(f"{output_dir = }")
        idx = output_dir.split("/")[-1]
        print("preprocessing ...\r", end='')

        if not is_stereo:
            img0, img0_depth = datas
            img0_depth = img0_depth.to(self.device)
        else:
            img0, img1, disp0 = datas
            img0_depth = Convert.disparity_to_depth(disp0).to(device)
        img0 = img0.to(self.device)
        img0_depth = utils.normalize_depth(img0_depth)
        disp0 = Convert.depth_to_disparity(img0_depth)
        flow01 = Convert.disparity_to_flow(disp0, device=self.device, random_sign=False)
        img0_all = torch.cat((img0, img0_depth, flow01 * -1.0), axis=0) 
        img0_all_fw, img1_valid, collision = self.fw(img0_all, flow01, img0_depth)
        img0_all_fw = img0_all_fw.squeeze(0)
        img1, img1_depth, back_flow01 = img0_all_fw[0:3], img0_all_fw[3:4], img0_all_fw[4:6]
        img1 = img1 * img1_valid
        img1_depth = img1_depth * img1_valid
        back_flow01 = back_flow01 * img1_valid
        img1_depth = utils.fix_warped_depth(img1_depth)
        img1 = utils.inpaint(img1, img1_valid, collision)

        assert torch.sum(torch.logical_and(img1_valid != 0, img1_valid != 1)) == 0

        _, h, w = img0.shape
        
        flow12, T1 = Convert.depth_to_random_flow(img1_depth, self.device)
        img1_all = torch.cat((img1, img1_depth, flow12 * -1.0, img1_valid), axis=0) 
        img1_all_fw, valid, collision = self.fw(img1_all, flow12, img1_depth)
        img1_all_fw = img1_all_fw.squeeze(0) 
        img2, img2_depth, back_flow12, fw_img1_valid = img1_all_fw[0:3], img1_all_fw[3:4], img1_all_fw[4:6], img1_all_fw[6:7]
        img2_valid = valid * fw_img1_valid
        img2 = img2 * img2_valid
        img2_depth = img2_depth * img2_valid
        back_flow12 = back_flow12 * img2_valid
        img2 = utils.inpaint(img2, img2_valid, collision)
        img2_depth = utils.fix_warped_depth(img2_depth)
        assert torch.sum(torch.logical_and(img2_valid != 0, img2_valid != 1)) == 0

        flow03, _ = Convert.depth_to_random_flow(img0_depth, self.device, T1=T1)
        img0_all = torch.cat((img0, img0_depth, flow03 * -1.0), axis=0) 
        img3_all_fw, img3_valid, collision = self.fw(img0_all, flow03, img0_depth)
        img3_all_fw = img3_all_fw.squeeze(0) 
        img3, img3_depth, back_flow03 = img3_all_fw[0:3], img3_all_fw[3:4], img3_all_fw[4:6]
        img3 = img3 * img3_valid
        img3_depth = img3_depth * img3_valid
        back_flow03 = back_flow03 * img3_valid
        img3 = utils.inpaint(img3, img3_valid, collision)
        img3_depth = utils.fix_warped_depth(img3_depth)
        assert torch.sum(torch.logical_and(img3_valid != 0, img3_valid != 1)) == 0

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        flow02, flow02_valid = self.cf(flow01, back_flow01, flow12, img1_depth)
        img0_all = torch.cat((img0, img0_depth, flow02 * -1.0, flow02_valid), axis=0) 
        img0_all_fw, valid, collision = self.fw(img0_all, flow02, img0_depth)
        img2_prime, img2_depth_prime, back_flow02_prime, fw_flow02_valid = img0_all_fw[0:3], img0_all_fw[3:4], img0_all_fw[4:6], img0_all_fw[6:7]

        img2_prime_valid = valid * fw_flow02_valid
        img2_prime = img2_prime * img2_prime_valid
        img2_depth_prime = img2_depth_prime * img2_prime_valid
        back_flow02_prime = back_flow02_prime * img2_prime_valid

        img2_prime = utils.inpaint(img2_prime, img2_prime_valid, collision)
        img2_depth_prime = utils.fix_warped_depth(img2_depth_prime)
        assert torch.sum(torch.logical_and(img2_prime_valid != 0, img2_prime_valid != 1)) == 0

        flow13, flow13_valid = self.cf(back_flow01, flow01, flow03, img1_depth)
        flow13_valid = flow13_valid * img1_valid
        img1_all = torch.cat((img1, img1_depth, flow13 * -1.0, flow13_valid), axis=0) 
        img1_all_fw, valid, collision = self.fw(img1_all, flow13, img1_depth)
        img3_prime, img3_depth_prime, back_flow13_prime, fw_flow13_valid = img1_all_fw[0:3], img1_all_fw[3:4], img1_all_fw[4:6], img1_all_fw[6:7]
        img3_prime_valid = valid * fw_flow13_valid
        img3_prime = img3_prime * img3_prime_valid
        img3_depth_prime = img3_depth_prime * img3_prime_valid
        back_flow13_prime = back_flow13_prime * img3_prime_valid
        img3_prime = utils.inpaint(img3_prime, img3_prime_valid, collision)
        img3_depth_prime = utils.fix_warped_depth(img3_depth_prime)
        assert torch.sum(torch.logical_and(img3_prime_valid != 0, img3_prime_valid != 1)) == 0

        group = [(img0, img0_depth, img1, img1_depth, flow01, back_flow01),
                 # 0:3,        3:4,  4:7,        7:8,   8:10,       10:12
                 (img1, img1_depth, img2, img2_depth, flow12, back_flow12),
                 (img0, img0_depth, img2_prime, img2_depth_prime, flow02, back_flow02_prime),
                 (img0, img0_depth, img3, img3_depth, flow03, back_flow03),
                 (img1, img1_depth, img3_prime, img3_depth_prime, flow13, back_flow13_prime)]
        for group_idx, pair in enumerate(group):
            to_save_data0 = torch.cat(pair, axis=0).cpu().numpy()
            assert to_save_data0.shape == (12, h, w), "wrong data shape"

        to_save_data0 = torch.cat((img0, img0_depth, img1, img1_depth, img2, img2_depth, img3, img3_depth,
                                   img2_prime, img2_depth_prime, img3_prime, img3_depth_prime, 
                                   flow01, back_flow01, flow12, back_flow12, flow02, back_flow02_prime, 
                                   flow03, back_flow03, flow13, back_flow13_prime), axis=0).cpu().numpy()

        if to_save_data0.shape != (44, h, w):
            print(f"something wrong when preprocess {output_dir}, {to_save_data0.shape = }")
            sys.exit()

        np.savez_compressed(f"{output_dir}/group.npz",
                            img_depth_flow=to_save_data0)

        end_preprocess_time = time.time()
        print(f"preprocessing time = {end_preprocess_time - start_preprocess_time}")
        
        start_augment_time = time.time()
        for group_idx, (imgA, imgA_depth, imgB, imgB_depth, flowAB, back_flowAB) in enumerate(group):
            for augment_idx, augment_flow_type in enumerate([0, 5, 6, 7, 1, 5, 6, 7, 2, 5, 6, 7]):
                print(f"                                        \r", end='')
                print(f"augmenting ... {group_idx}_{augment_idx}\r", end='')

                set1, set2, _, set3 = augment_flow(imgA, imgA_depth, imgB, imgB_depth, flowAB,
                                                   back_flowAB, device=self.device,
                                                   augment_flow_type=augment_flow_type)

                to_save_data1 = torch.cat([set1[i] for i in range(0, 4)], axis=0)
                to_save_data2 = torch.cat([set2[i] for i in range(2, 6)], axis=0

                to_save_data1_np = to_save_data1.detach().cpu().numpy()
                to_save_data2_np = to_save_data2.detach().cpu().numpy()

                assert to_save_data1_np.shape == (8, h, w), "wrong augmented data1"
                assert to_save_data2_np.shape == (8, h, w), "wrong augmented data2"

                np.savez_compressed(f"{output_dir}/{group_idx}_{augment_idx}_1.npz",
                                    img_depth_flow=to_save_data1_np,
                                    augment_flow_type=augment_flow_type)
                np.savez_compressed(f"{output_dir}/{group_idx}_{augment_idx}_2.npz",
                                    img_depth_flow=to_save_data2_np,
                                    augment_flow_type=augment_flow_type)
                

        end_augment_time = time.time()
        print(f"augmenting time = {end_augment_time - start_augment_time}")

        del augment_idx, augment_flow_type
        del imgA, imgB, imgA_depth, imgB_depth
        del flowAB, back_flowAB
        del set1, set2
        del to_save_data1, to_save_data2
        del to_save_data1_np, to_save_data2_np

        del img0, img1, img2
        del img0_depth, img1_depth, img2_depth
        del disp0

        if is_stereo:
            del disp1

        del img1_all, img1_all_fw
        del flow01, flow12, flow02
        del back_flow01, back_flow12
        del img2_prime, img2_depth_prime, back_flow02_prime
        del img3_prime, img3_depth_prime, back_flow13_prime
        del output_dir

        del start_augment_time, end_augment_time
        del start_preprocess_time, end_preprocess_time

        del to_save_data0

def read_args():
    parser = ArgumentParser()

    parser.add_argument('--dataset')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--split', default=1, type=int)
    parser.add_argument('--split_id', default=0, type=int)
    parser.add_argument('--specific_epoch_idx', default=-1, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()

    seed_mapping = {"DIML": 30, "ReDWeb": 40}
    utils.set_seed(12345 + seed_mapping[args.dataset])
    print(f"{12345 + seed_mapping[args.dataset] = }")
    print(args)

    if args.dataset == "DIML":
        dataset = DIML()
        output_dir = f"datasets/AugmentedDatasets/DIML"
        is_stereo = True
    elif args.dataset == "ReDWeb":
        dataset = ReDWeb()
        output_dir = "datasets/AugmentedDatasets/ReDWeb"
        is_stereo = False
    else:
        print(f"{args.dataset = }")

    device = f"cuda:{args.gpu}"
    ppa = PreprocessPlusAugment(device=device)

    split_len = int((len(dataset) + args.split - 1) // args.split)
    start = int(args.split_id * split_len)
    end = int((args.split_id + 1) * split_len)
    if args.split_id == args.split - 1:
        end = len(dataset)

    data = dataset[0]
    print(f"{len(dataset) = }, {is_stereo = }, {start = }, {end = }")

    for epoch_idx in range(2):
        for img_idx in range(start, end):
            print(f"{img_idx + 1} / {len(dataset)}: {epoch_idx = }, seed = {12345 + img_idx + epoch_idx * len(dataset)}")
            utils.set_seed(12345 + img_idx + epoch_idx * len(dataset))
            datas = dataset[img_idx]

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            ppa(datas, f"{output_dir}/{img_idx + epoch_idx * len(dataset)}", is_stereo)
