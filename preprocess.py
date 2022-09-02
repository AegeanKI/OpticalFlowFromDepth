import utils
# import using_clib
# from using_clib import Resample2d
# from using_clib import Resample2d2
import geometry  
import flow_colors
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import ctypes
import matplotlib.pyplot as plt
from bilateral_filter import sparse_bilateral_filtering
import argparse
import time
import skimage
import glob
import os

from cuda.fw import FW
from multiprocessing import Pool

same_range = 1.0

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
                           [0,       0,   0, 1]]], dtype=torch.float64)

        if another:
            K[:, :2, :2] *= 2

        K[:, 0, :] *= w
        K[:, 1, :] *= h
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

        axisangle = torch.tensor([[camera_ang]], dtype=torch.float64)
        translation = torch.tensor([[camera_mot]], dtype=torch.float64)

        if another_axisangle is not None and another_translation is not None:
            T = geometry.transformation_from_parameters(axisangle + another_axisangle,
                                                        translation + another_translation)
        else:
            T = geometry.transformation_from_parameters(axisangle, translation)
        return T, axisangle, translation


class Convert():
    @staticmethod
    def depth_to_disparity(depth):
        _, _, _, w = depth.shape
        s = utils.get_random(175, 50, random_sign=False)
        
        depth_max = torch.max(depth)
        disparity = (1. / depth) * s * depth_max / w
        # B, f = Plausible.B(), Plausible.f()
        # disparity = B * f / depth
        return disparity

    @staticmethod
    def disparity_to_flow(disparity, random_sign=True):
        _, _, h, w = disparity.shape

        flow = torch.zeros_like(disparity).repeat(1, 2, 1, 1)
        flow[:, 0:1, :, :] = disparity
        if random_sign:
            flow = flow * utils.get_random(0, 1)
        return flow

    @staticmethod
    def disparity_to_depth(disparity):
        B, f = Plausible.B(), Plausible.f()
        depth = B * f / (disparity + 0.25)
        depth[depth > 100] = 100
        return depth

    @staticmethod
    def depth_to_random_flow(depth, segment=None):
        b, _, h, w = depth.shape
        K, inv_K = Plausible.K((h, w))
        K, inv_K = K.to("cuda"), inv_K.to("cuda")
        backproject_depth = geometry.BackprojectDepth(b, h, w)
        project_3d = geometry.Project3D(b, h, w)

        cam_points = backproject_depth(depth, inv_K)

        T1, axisangle, translation = Plausible.random_motion(1. / 144., 1. / 144., 0.15, 0.15) 
        T1 = T1.to("cuda")
        p1, z1 = project_3d(cam_points, K, T1)
        z1 = z1.reshape(b, h, w)
        
        if segment is not None:
            instances, labels = segment
            for l in range(len(labels)):
                Ti, _, _ = Plausible.random_motion(1. / 108., 1. / 108., 0.02, 0.02,
                                                    axisangle, translation) 
                pi, zi = project_3d(cam_points, K, Ti)
                zi = zi.reshape(b, h, w)
                p1[instances[l] > 0] = pi[instances[l] > 0]
                z1[instances[l][:, :, :, 0] > 0] = zi[instances[l][:, :, :, 0] > 0]

        p1 = (p1 + 1) / 2
        p1[:, :, :, 0] *= w - 1
        p1[:, :, :, 1] *= h - 1
        p1 = p1.double()

        meshgrid = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        p0 = torch.stack(meshgrid, axis=-1).type(torch.float64).to("cuda")
        flow = p1 - p0
        flow = flow.permute(0, 3, 1, 2)
        return flow


class ConcatFlow(nn.Module):
    def __init__(self, size, batch_size):
        super().__init__()
        self.fw = FW(size, batch_size).to("cuda")
    
    def forward(self, flowAB, back_flowAB, flowBC, imgB_depth):
        concat_flow = self.fw(flowBC, back_flowAB, imgB_depth)
        concat_flow = concat_flow + flowAB
        return concat_flow

class BackFlow(nn.Module):
    def __init__(self, size, batch_size):
        super().__init__()
        self.fw = FW(size, batch_size).to("cuda")

    def forward(self, flowAB, imgA_depth):
        back_flow = self.fw(flowAB, flowAB, imgA_depth)
        back_flow = back_flow * -1.
        return back_flow


class Preprocess(nn.Module):
    def __init__(self, batch_size, device='cuda'):
        super().__init__()
        self.device = device
        self.batch_size = batch_size

    def forward(self, img_names, mono):
        if mono:
            def get_img_and_depth_and_size(img_name):
                img_depth_name = img_name.replace("jpg", "png")
                img_full_path = f"input/monocular/image/{img_name}"
                depth_full_path = f"input/monocular/depth/{img_depth_name}"
                img, size = utils.get_img(img_full_path)
                img_depth  = utils.get_depth(depth_full_path, size, 16) # depth.png: 0 ~ 65535
                img_depth = utils.fix_depth(img_depth, img)
                return img, img_depth, size
        
            def stack_imgs_and_depths(imgs_and_depths_and_sizes):
                max_h, max_w = 0, 0
                for img, depth, (h, w) in imgs_and_depths_and_sizes:
                    max_h = max(max_h, h)
                    max_w = max(max_w, w)

                img0 = torch.zeros((self.batch_size, 3, max_h, max_w)).type(torch.float64).to("cuda")
                img0_depth = torch.ones((self.batch_size, 1, max_h, max_w)).type(torch.float64).to("cuda") * 100.
                for idx, (img, depth, size) in enumerate(imgs_and_depths_and_sizes):
                    h, w = size
                    img0[idx, :, :h, :w] = img
                    img0_depth[idx, :, :h, :w] = utils.fix_depth(depth, img)
                return img0, img0_depth, (max_h, max_w)

            imgs_and_depths_and_sizes = [get_img_and_depth_and_size(img_name) for img_name in img_names]
            img0, img0_depth, size = stack_imgs_and_depths(imgs_and_depths_and_sizes)
            disparity = Convert.depth_to_disparity(img0_depth)
        else:
            img0_full_path = f"input/stereo/image_2/{img0_name}"
            img1_full_path = f"input/stereo/image_3/{img0_name}"
            disparity_full_path = f"input/stereo/disparity/{img0_name}"
            img0, img1_real, size = utils.get_stereo_img(img0_full_path, img1_full_path)

            disparity = utils.get_disparity(disparity_full_path, size)
            img0_depth = Convert.disparity_to_depth(disparity)
            img0_depth = utils.fix_depth(img0_depth, img0)
        
        fw = FW(size, self.batch_size).to("cuda")

        flow01 = Convert.disparity_to_flow(disparity, random_sign=mono)
        img0_all = torch.cat((img0, img0_depth, flow01 * -1), axis=1) 
        img0_all_fw = fw(img0_all, flow01, img0_depth) 
        img1, img1_depth, back_flow01 = img0_all_fw[:, 0:3], img0_all_fw[:, 3:4], img0_all_fw[:, 4:6]
        img1 = img1 if mono else img1_real
        img1_depth = utils.fix_depth(img1_depth, img1)

        flow12 = Convert.depth_to_random_flow(img1_depth)
        img1_all = torch.cat((img1, img1_depth, flow12 * -1), axis=1) 
        img1_all_fw = fw(img1_all, flow12, img1_depth) 
        img2, img2_depth, back_flow12 = img1_all_fw[:, 0:3], img1_all_fw[:, 3:4], img1_all_fw[:, 4:6]
        img2_depth = utils.fix_depth(img2_depth, img2)

        cf = ConcatFlow(size, self.batch_size).to("cuda")
        bf = BackFlow(size, self.batch_size).to("cuda")
        flow02 = cf(flow01, back_flow01, flow12, img1_depth)
        back_flow02 = bf(flow02, img0_depth)

        concat_img = fw(img0, flow02, img0_depth)

        threads = []
        for idx, (img_name, (_, _, size)) in enumerate(zip(img_names, imgs_and_depths_and_sizes)):
            thread = MultiThreadSave(mono, idx, img_name, size, img0, img1, img2, img0_depth,
                                     img1_depth, img2_depth, concat_img, flow01, flow12, flow02,
                                     back_flow01, back_flow12, back_flow02)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()


from threading import Thread
class MultiThreadSave(Thread):
    def __init__(self, mono, idx, img_name, size, img0, img1, img2, img0_depth, img1_depth, img2_depth,
                 concat_img, flow01, flow12, flow02, back_flow01, back_flow12, back_flow02):
        super().__init__()
        self.mono = mono
        self.idx = idx
        self.img_name = img_name
        self.size = size
        self.img0 = img0
        self.img1 = img1
        self.img2 = img2
        self.img0_depth = img0_depth
        self.img1_depth = img1_depth
        self.img2_depth = img2_depth
        self.concat_img = concat_img
        self.flow01 = flow01
        self.flow12 = flow12
        self.flow02 = flow02
        self.back_flow01 = back_flow01
        self.back_flow12 = back_flow12
        self.back_flow02 = back_flow02

    def run(self):
        if self.mono:
            output_dir = f"output/monocular/{self.img_name}"
        else:
            output_dir = f"output/stereo/{self.img_name}"
        # output_dir = f"testing/{self.img_name}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        h, w = self.size
        np.save(f"{output_dir}/img0.npy", self.img0[self.idx, :, :h, :w].cpu().numpy())
        np.save(f"{output_dir}/img0_depth.npy", self.img0_depth[self.idx, :, :h, :w].cpu().numpy())
        np.save(f"{output_dir}/img1.npy", self.img1[self.idx, :, :h, :w].cpu().numpy())
        np.save(f"{output_dir}/img1_depth.npy", self.img1_depth[self.idx, :, :h, :w].cpu().numpy())
        np.save(f"{output_dir}/img2.npy", self.img2[self.idx, :, :h, :w].cpu().numpy())
        np.save(f"{output_dir}/img2_depth.npy", self.img2_depth[self.idx, :, :h, :w].cpu().numpy())

        np.save(f"{output_dir}/concat_img.npy", self.concat_img[self.idx, :, :h, :w].cpu().numpy())
        
        np.save(f"{output_dir}/flow01.npy", self.flow01[self.idx, :, :h, :w].cpu().numpy())
        np.save(f"{output_dir}/back_flow01.npy", self.back_flow01[self.idx, :, :h, :w].cpu().numpy())
        np.save(f"{output_dir}/flow12.npy", self.flow12[self.idx, :, :h, :w].cpu().numpy())
        np.save(f"{output_dir}/back_flow12.npy", self.back_flow12[self.idx, :, :h, :w].cpu().numpy())
        np.save(f"{output_dir}/flow02.npy", self.flow02[self.idx, :, :h, :w].cpu().numpy())
        np.save(f"{output_dir}/back_flow02.npy", self.back_flow02[self.idx, :, :h, :w].cpu().numpy())



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono", type=int, default=1, help="use monocular or not")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    start = time.time()

    if args.mono:
        fp = open('dCOCO_file_list.txt', 'r')
        img_names = fp.readlines()
        fp.close()
        if not os.path.exists("output"):
            os.mkdir("output")
        if not os.path.exists("output/monocular"):
            os.mkdir("output/monocular")
    else:
        img_names = glob.glob(f"input/stereo/image_2/*")
    num_imgs = len(img_names)

    img_names = np.array(img_names).reshape((-1, args.batch_size))

    preprocess = Preprocess(args.batch_size).to('cuda')

    for idx, img_names in enumerate(img_names):
        img_names = [img_name[:-1] for img_name in img_names]
        print(f"{idx * args.batch_size + 1} ~ {(idx + 1) * args.batch_size} / {num_imgs}")
        preprocess(img_names, True)

    end = time.time()
    print(f"cal time = {end - start}")
    torch.cuda.empty_cache()
