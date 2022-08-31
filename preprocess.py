import utils
# import using_clib
from using_clib import Resample2d
from using_clib import Resample2d2
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
        K = np.array([[[0.58,    0, 0.5, 0],
                       # [0,    1.92, 0.5, 0],
                       [0,    0.58, 0.5, 0],
                       [0,       0,   1, 0], 
                       [0,       0,   0, 1]]], dtype=np.float64)

        if another:
            K[:, :2, :2] *= 2

        K[:, 0, :] *= w
        K[:, 1, :] *= h
        return torch.from_numpy(K), torch.from_numpy(np.linalg.pinv(K))

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

        axisangle = torch.from_numpy(np.array([[camera_ang]], dtype=np.float64))
        translation = torch.from_numpy(np.array([[camera_mot]]))

        if another_axisangle is not None and another_translation is not None:
            T = geometry.transformation_from_parameters(axisangle + another_axisangle,
                                                        translation + another_translation)
        else:
            T = geometry.transformation_from_parameters(axisangle, translation)
        return T, axisangle, translation


class Convert():
    @staticmethod
    def depth_to_disparity(depth, size):
        h, w = size
        s = utils.get_random(175, 50, random_sign=False)
        
        depth_max = np.max(depth)
        disparity = (1. / depth) * s * depth_max / w
        # B, f = Plausible.B(), Plausible.f()
        # disparity = B * f / depth
        return disparity

    @staticmethod
    def disparity_to_flow(disparity, size, random_sign=True):
        h, w = size
        flow = np.zeros((1, h, w, 2))
        flow[0, :, :, 0] = -disparity
        if random_sign:
            flow = flow * utils.get_random(0, 1)
        flow = torch.from_numpy(flow).double()
        return flow

    @staticmethod
    def disparity_to_depth(disparity):
        B, f = Plausible.B(), Plausible.f()
        depth = B * f / (disparity + 0.25)
        depth[depth > 100] = 100
        return depth

    @staticmethod
    def depth_to_random_flow(depth, size, segment=None):
        h, w = size
        K, inv_K = Plausible.K(size)
        backproject_depth = geometry.BackprojectDepth(1, h, w)
        project_3d = geometry.Project3D(1, h, w)

        depth = torch.from_numpy(depth).unsqueeze(0).float()
        cam_points = backproject_depth(depth, inv_K)

        T1, axisangle, translation = Plausible.random_motion(1. / 144., 1. / 144., 0.05, 0.05) 
        p1, z1 = project_3d(cam_points, K, T1)
        z1 = z1.reshape(1, h, w)
        
        if segment is not None:
            instances, labels = segment
            for l in range(len(labels)):
                Ti, _, _ = Plausible.random_motion(1. / 108., 1. / 108., 0.02, 0.02,
                                                    axisangle, translation) 
                pi, zi = project_3d(cam_points, K, Ti)
                zi = zi.reshape(1, h, w)
                p1[instances[l] > 0] = pi[instances[l] > 0]
                z1[instances[l][:, :, :, 0] > 0] = zi[instances[l][:, :, :, 0] > 0]


        p1 = (p1 + 1) / 2
        p1[:, :, :, 0] *= w - 1
        p1[:, :, :, 1] *= h - 1
        p1 = p1.double()

        meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
        p0 = np.stack(meshgrid, axis=-1).astype(np.float64)   
        flow = p1 - p0
        return flow

    @staticmethod
    def flow_to_backward_flow(flow, depth, size):
        flow_np = flow[0].detach().numpy()
        backward_flow = Resample2d()(flow_np, flow, depth, size)
        backward_flow = torch.from_numpy(backward_flow).unsqueeze(0) * -1.
        return backward_flow

    @staticmethod
    def two_contiguous_flows_to_one_flow(flow1, backward_flow1, flow2, mid_depth, size):
        flow2_np = flow2[0].detach().numpy()
        concat_flow = Resample2d()(flow2_np, backward_flow1, mid_depth, size)
        concat_flow = torch.from_numpy(concat_flow).unsqueeze(0)
        concat_flow = concat_flow + flow1
        return concat_flow

class ConcatFlow(nn.Module):
    def __init__(self, size, batch_size):
        super().__init__()
        self.size = size
        self.batch_size = batch_size
        self.fw = Resample2d2(size, batch_size)
    
    def forward(self, flowAB, back_flowAB, flowBC, imgB_depth):
        concat_flow = self.fw(flowBC, back_flowAB, imgB_depth)
        concat_flow = concat_flow + flowAB
        return concat_flow


class Preprocess(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.fw = Resample2d()

    def forward(self, img0_name, mono):
        if mono:
            output_dir = f"output/monocular/{img0_name}"
        else:
            output_dir = f"output/stereo/{img0_name}"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        print(f"{img0_name} {idx + 1} / {num_imgs}")

        if mono:
            img0_full_path = f"input/monocular/image/{img0_name}"
            depth_full_path = f"input/monocular/depth/{img0_name}"
            depth_full_path = depth_full_path.replace("jpg", "png")
            img0, size = utils.get_img(img0_full_path)
            img0_depth  = utils.get_depth(depth_full_path, size, 16) # depth.png: 0 ~ 65535
            img0_depth = utils.fix_depth(img0_depth, img0)
            disparity = Convert.depth_to_disparity(img0_depth, size)
            flow01 = Convert.disparity_to_flow(disparity, size)
        else:
            img0_full_path = f"input/stereo/image_2/{img0_name}"
            img1_full_path = f"input/stereo/image_3/{img0_name}"
            disparity_full_path = f"input/stereo/disparity/{img0_name}"
            img0, img1, size = utils.get_stereo_img(img0_full_path, img1_full_path)
            disparity = utils.get_disparity(disparity_full_path, size)
            img0_depth = Convert.disparity_to_depth(disparity)
            img0_depth = utils.fix_depth(img0_depth, img0)
            flow01 = Convert.disparity_to_flow(disparity, size, False)

        np.save(f"{output_dir}/img0.npy", img0)
        np.save(f"{output_dir}/img0_depth.npy", img0_depth)
        # cv2.imwrite(f"{output_dir}/img0.png", img0)
        # plt.imsave(f"{output_dir}/img0_depth.png", 1 / img0_depth, cmap="magma")
        # print(f"{np.min(img0) = }")
        # print(f"{np.max(img0) = }")
        # print(f"{np.min(img0_depth) = }")
        # print(f"{np.max(img0_depth) = }")

        back_flow01 = Convert.flow_to_backward_flow(flow01, img0_depth, size)
        # _, flow01_color = utils.color_flow(flow01)
        # _, back_flow01_color = utils.color_flow(back_flow01)
        torch.save(flow01, f"{output_dir}/flow01.pt")
        torch.save(back_flow01, f"{output_dir}/back_flow01.pt")
        # cv2.imwrite(f"{output_dir}/flow01.png", flow01_color)
        # cv2.imwrite(f"{output_dir}/back_flow01.png", back_flow01_color)
        # print(f"{torch.min(flow01) = }")
        # print(f"{torch.max(flow01) = }")
        # print(f"{torch.min(back_flow01) = }")
        # print(f"{torch.max(back_flow01) = }")


        if mono:
            img1 = self.fw(img0, flow01, img0_depth, size)
        img1_depth = self.fw(img0_depth, flow01, img0_depth, size)
        img1_depth = utils.fix_depth(img1_depth, img1)
        np.save(f"{output_dir}/img1.npy", img1)
        np.save(f"{output_dir}/img1_depth.npy", img1_depth)
        # cv2.imwrite(f"{output_dir}/img1.png", img1)
        # plt.imsave(f"{output_dir}/img1_depth.png", 1 / img1_depth, cmap="magma")
        # print(f"{np.min(img1) = }")
        # print(f"{np.max(img1) = }")
        # print(f"{np.min(img1_depth) = }")
        # print(f"{np.max(img1_depth) = }")


        flow12 = Convert.depth_to_random_flow(img1_depth, size)
        back_flow12 = Convert.flow_to_backward_flow(flow12, img1_depth, size)
        # _, flow12_color = utils.color_flow(flow12)
        # _, back_flow12_color = utils.color_flow(back_flow12)
        torch.save(flow12, f"{output_dir}/flow12.pt")
        torch.save(back_flow12, f"{output_dir}/back_flow12.pt")
        # cv2.imwrite(f"{output_dir}/flow12.png", flow12_color)
        # cv2.imwrite(f"{output_dir}/back_flow12.png", back_flow12_color)
        # print(f"{torch.min(flow12) = }")
        # print(f"{torch.max(flow12) = }")
        # print(f"{torch.min(back_flow12) = }")
        # print(f"{torch.max(back_flow12) = }")

        img2 = self.fw(img1, flow12, img1_depth, size)
        img2_depth = self.fw(img1_depth, flow12, img1_depth, size)
        img2_depth = utils.fix_depth(img2_depth, img1)
        np.save(f"{output_dir}/img2.npy", img2)
        np.save(f"{output_dir}/img2_depth.npy", img2_depth)
        # cv2.imwrite(f"{output_dir}/img2.png", img2)
        # plt.imsave(f"{output_dir}/img2_depth.png", 1 / img2_depth, cmap="magma")
        # print(f"{np.min(img2) = }")
        # print(f"{np.max(img2) = }")
        # print(f"{np.min(img2_depth) = }")
        # print(f"{np.max(img2_depth) = }")

        flow02 = Convert.two_contiguous_flows_to_one_flow(flow01, back_flow01, flow12,
                                                          img1_depth, size)
        back_flow02 = Convert.flow_to_backward_flow(flow02, img0_depth, size)
        # _, flow02_color = utils.color_flow(flow02)
        # _, back_flow02_color = utils.color_flow(back_flow02)
        torch.save(flow02, f"{output_dir}/flow02.pt")
        torch.save(back_flow02, f"{output_dir}/back_flow02.pt")
        # cv2.imwrite(f"{output_dir}/flow02.png", flow02_color)
        # cv2.imwrite(f"{output_dir}/back_flow02.png", back_flow02_color)
        # print(f"{torch.min(flow02) = }")
        # print(f"{torch.max(flow02) = }")
        # print(f"{torch.min(back_flow02) = }")
        # print(f"{torch.max(back_flow02) = }")




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono", type=int, default=0, help="use monocular or not")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    start = time.time()

    if args.mono:
        fp = open('dCOCO_file_list.txt', 'r')
        img_names = fp.readlines()
        fp.close()
    else:
        img_names = glob.glob(f"input/stereo/image_2/*")
    num_imgs = len(img_names)

    preprocess = Preprocess().to('cuda')

    for idx, img0_name in enumerate(img_names):
        img0_name = img0_name[:-1]
        preprocess(img0_name, args.mono)

    end = time.time()
    print(f"cal time = {end - start}")
    torch.cuda.empty_cache()
