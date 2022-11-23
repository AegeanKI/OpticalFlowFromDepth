import utils
# import using_clib
# from using_clib import Resample2d
# from using_clib import Resample2d2
import geometry  
# import flow_colors
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
# import ctypes
# import matplotlib.pyplot as plt
from bilateral_filter import sparse_bilateral_filtering
import argparse
import time
# import skimage
import glob
import os

from my_cuda.fw import FW
from threading import Thread
from multiprocessing import Pool
import gc
import matplotlib.pyplot as plt

# same_range = 1.0

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
        _, _, _, w = depth.shape
        s = utils.get_random(175, 50, random_sign=False)
        
        depth_max = torch.max(depth)
        disparity = (1. / depth) * s * depth_max / w
        # B, f = Plausible.B(), Plausible.f()
        # disparity = B * f / depth
        del w, s, depth_max
        return disparity

    @staticmethod
    def disparity_to_flow(disparity, random_sign=True):
        _, _, h, w = disparity.shape

        flow = torch.zeros_like(disparity).repeat(1, 2, 1, 1)
        flow[:, 0:1, :, :] = disparity
        if random_sign:
            flow = flow * utils.get_random(0, 1)

        del h, w
        return flow

    @staticmethod
    def disparity_to_depth(disparity):
        B, f = Plausible.B(), Plausible.f()
        depth = B * f / (disparity + 0.25)
        depth[depth > 100] = 100

        del B, f
        return depth

    @staticmethod
    def depth_to_random_flow(depth, device="cuda", segment=None):
        b, _, h, w = depth.shape
        K, inv_K = Plausible.K((h, w))
        K, inv_K = K.to(device), inv_K.to(device)
        backproject_depth = geometry.BackprojectDepth(b, h, w)
        project_3d = geometry.Project3D(b, h, w)

        with torch.no_grad():
            cam_points = backproject_depth(depth, inv_K)

        T1, axisangle, translation = Plausible.random_motion(1. / 36., 1. / 36., 0.1, 0.1) 
        T1 = T1.to(device)

        with torch.no_grad():
            p1, z1 = project_3d(cam_points, K, T1)
        z1 = z1.reshape(b, h, w)
            
        if segment is not None:
            instances, labels = segment
            for l in range(len(labels)):
                Ti, _, _ = Plausible.random_motion(1. / 72., 1. / 72., 0.05, 0.05,
                                                    axisangle, translation) 
                pi, zi = project_3d(cam_points, K, Ti)
                zi = zi.reshape(b, h, w)
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

        return flow


class ConcatFlow(nn.Module):
    def __init__(self, size, batch_size, device="cuda"):
        super().__init__()
        self.device = device
        self.fw = FW(size, batch_size, device).to(device)
    
    def forward(self, flowAB, back_flowAB, flowBC, imgB_depth):
        with torch.no_grad():
            concat_flow = self.fw(flowBC, back_flowAB, imgB_depth)
        concat_flow = concat_flow + flowAB
        return concat_flow.to(self.device)

class BackFlow(nn.Module):
    def __init__(self, size, batch_size, device="cuda"):
        super().__init__()
        self.device = device
        self.fw = FW(size, batch_size, device).to(device)

    def forward(self, flowAB, imgA_depth):
        with torch.no_grad():
            back_flow = self.fw(flowAB, flowAB, imgA_depth)
        back_flow = back_flow * -1.
        return back_flow.to(self.device)


class PreprocessStereo(nn.Module):
    def __init__(self, batch_size, device='cuda'):
        super().__init__()
        self.device = device
        self.batch_size = batch_size

    def forward(self, batch_idx, img0, img1, disparity, output_dir):
        # disparity = Convert.depth_to_disparity(img0_depth)
        start_warping_time = time.time()

        img0_depth = Convert.disparity_to_depth(disparity)
        img0_depth = utils.multithread_fix_depth(img0_depth, img0)

        _, _, h, w = img0.shape
        size = (h, w)

        fw = FW(size, self.batch_size, self.device).to(self.device)
        cf = ConcatFlow(size, self.batch_size).to(self.device)
        bf = BackFlow(size, self.batch_size).to(self.device)

        flow01 = Convert.disparity_to_flow(disparity, random_sign=False)
        img0_all = torch.cat((img0, img0_depth, flow01 * -1), axis=1) 
        with torch.no_grad():
            img0_all_fw = fw(img0_all, flow01, img0_depth) 
        img1, img1_depth, back_flow01 = img0_all_fw[:, 0:3], img0_all_fw[:, 3:4], img0_all_fw[:, 4:6]
        img1_depth = utils.multithread_fix_depth(img1_depth, img1)

        flow12 = Convert.depth_to_random_flow(img1_depth, self.device)
        img1_all = torch.cat((img1, img1_depth, flow12 * -1), axis=1) 

        with torch.no_grad():
            img1_all_fw = fw(img1_all, flow12, img1_depth) 
        img2, img2_depth, back_flow12 = img1_all_fw[:, 0:3], img1_all_fw[:, 3:4], img1_all_fw[:, 4:6]
        img2_depth = utils.multithread_fix_depth(img2_depth, img2)

        with torch.no_grad():
            flow02 = cf(flow01, back_flow01, flow12, img1_depth)
            back_flow02 = bf(flow02, img0_depth)

        with torch.no_grad():
            concat_img = fw(img0, flow02, img0_depth)

        end_warping_time = time.time()
        print(f"warping time = {end_warping_time - start_warping_time}")
        # start_concat_time = time.time()

        to_save_data = torch.cat((img0, img1, img2, img0_depth, img1_depth, img2_depth, concat_img,
                                  flow01, flow12, flow02, back_flow01, back_flow12, back_flow02), axis=1)

        # end_concat_time = time.time()
        # print(f"concat time = {end_concat_time - start_concat_time}")
        start_saving_time = time.time()

        # utils.multithread_save_img_and_img_depth_and_flow(
        #     img_names, original_size, output_dir, img0, img1, img2, img0_depth,
        #     img1_depth, img2_depth, concat_img, flow01, flow12, flow02,
        #     back_flow01, back_flow12, back_flow02)
        # utils.multithread_save_data(img_names, original_size, output_dir, to_save_data)
        # utils.batch_save_data(batch_idx, output_dir, original_size, to_save_data)
        utils.batch_save_data(batch_idx, output_dir, to_save_data)

        end_saving_time = time.time()
        print(f"saving time = {end_saving_time - start_saving_time}")

        del img0, img1, img2, concat_img
        del img0_depth, img1_depth, img2_depth
        del img0_all, img1_all, img0_all_fw, img1_all_fw
        del flow01, flow12, flow02
        del back_flow01, back_flow12, back_flow02
        # del res, disparity
        del disparity
        # del img_names, to_save_data, original_size
        del to_save_data
        del fw, cf, bf
        del output_dir

        del start_saving_time, end_saving_time
        # del start_get_img_time, end_get_img_time
        del start_warping_time, end_warping_time
        # del start_concat_time, end_concat_time




class PreprocessMono(nn.Module):
    def __init__(self, batch_size, device='cuda'):
        super().__init__()
        self.device = device
        self.batch_size = batch_size

    def forward(self, batch_idx, img0, img0_depth, output_dir):
        start_warping_time = time.time()

        disparity = Convert.depth_to_disparity(img0_depth)

        _, _, h, w = img0.shape
        size = (h, w)

        fw = FW(size, self.batch_size, self.device).to(self.device)
        cf = ConcatFlow(size, self.batch_size).to(self.device)
        bf = BackFlow(size, self.batch_size).to(self.device)

        flow01 = Convert.disparity_to_flow(disparity)
        img0_all = torch.cat((img0, img0_depth, flow01 * -1), axis=1) 
        with torch.no_grad():
            img0_all_fw = fw(img0_all, flow01, img0_depth) 
        img1, img1_depth, back_flow01 = img0_all_fw[:, 0:3], img0_all_fw[:, 3:4], img0_all_fw[:, 4:6]
        img1_depth = utils.multithread_fix_depth(img1_depth, img1)

        flow12 = Convert.depth_to_random_flow(img1_depth, self.device)
        img1_all = torch.cat((img1, img1_depth, flow12 * -1), axis=1) 

        with torch.no_grad():
            img1_all_fw = fw(img1_all, flow12, img1_depth) 
        img2, img2_depth, back_flow12 = img1_all_fw[:, 0:3], img1_all_fw[:, 3:4], img1_all_fw[:, 4:6]
        img2_depth = utils.multithread_fix_depth(img2_depth, img2)

        with torch.no_grad():
            flow02 = cf(flow01, back_flow01, flow12, img1_depth)
            back_flow02 = bf(flow02, img0_depth)

        with torch.no_grad():
            concat_img = fw(img0, flow02, img0_depth)

        end_warping_time = time.time()
        print(f"warping time = {end_warping_time - start_warping_time}")
        # start_concat_time = time.time()

        to_save_data = torch.cat((img0, img1, img2, img0_depth, img1_depth, img2_depth, concat_img,
                                  flow01, flow12, flow02, back_flow01, back_flow12, back_flow02), axis=1)

        # end_concat_time = time.time()
        # print(f"concat time = {end_concat_time - start_concat_time}")
        start_saving_time = time.time()

        # utils.multithread_save_img_and_img_depth_and_flow(
        #     img_names, original_size, output_dir, img0, img1, img2, img0_depth,
        #     img1_depth, img2_depth, concat_img, flow01, flow12, flow02,
        #     back_flow01, back_flow12, back_flow02)
        # utils.multithread_save_data(img_names, original_size, output_dir, to_save_data)
        # utils.batch_save_data(batch_idx, output_dir, original_size, to_save_data)
        utils.batch_save_data(batch_idx, output_dir, to_save_data)

        end_saving_time = time.time()
        print(f"saving time = {end_saving_time - start_saving_time}")

        del img0, img1, img2, concat_img
        del img0_depth, img1_depth, img2_depth
        del img0_all, img1_all, img0_all_fw, img1_all_fw
        del flow01, flow12, flow02
        del back_flow01, back_flow12, back_flow02
        # del res, disparity
        del disparity
        # del img_names, to_save_data, original_size
        del to_save_data
        del fw, cf, bf
        del output_dir

        del start_saving_time, end_saving_time
        # del start_get_img_time, end_get_img_time
        del start_warping_time, end_warping_time
        # del start_concat_time, end_concat_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono", type=int, default=0, help="use monocular or not")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    start = time.time()
    args = parse_args()

    torch.cuda.empty_cache()
    gc.collect()

    every_batch_change_dir = 320 // args.batch_size
    device = "cuda" if torch.cuda.is_available() else cpu()
    preprocess_mono = PreprocessMono(args.batch_size, device).to(device)
    preprocess_stereo = PreprocessStereo(args.batch_size, device).to(device)

    if args.mono:
        fp = open('dCOCO_file_list.txt', 'r')
        img_names = fp.readlines()
        fp.close()
    else:
        img_names = glob.glob(f"input/stereo/training/image_2/*")

    num_imgs = len(img_names)
    # img_names = np.array(img_names).reshape((-1, args.batch_size))

    if args.mono:
        os.mkdir(f"output/monocular")
    else:
        os.mkdir(f"output/stereo")


    img0 = None
    img1 = None
    img0_depth = None
    disparity = None
    batch_idx = 0
    # batch_size = 4
    # for batch_idx, img_names in enumerate(img_names):
    for _, img_name in enumerate(img_names):
        print(f"{img_name = }")
        if args.mono:
            img_name = img_name[:-1]
            img_full_path = f"input/monocular/image/{img_name}"
            print(f"{img_full_path = }")
            img, size = utils.get_img(img_full_path)
            if size[0] != 640 or size[1] != 480:
                continue

            depth_name = img_name.replace("jpg", "png")
            depth_full_path = f"input/monocular/depth/{depth_name}"
            depth = utils.get_depth(depth_full_path, size, 16)
            # depth = utils.multithread_fix_depth(depth, img)

            depth = depth.unsqueeze(0)
            img = img.unsqueeze(0)
            if img0 is None:
                img0 = img
                img0_depth = depth
            else:
                img0 = torch.cat((img0, img), axis=0)
                img0_depth = torch.cat((img0_depth, depth), axis=0)
        else:
            if img_name[-5] == '1':
                continue
            imgA_full_path = img_name
            imgB_full_path = imgA_full_path.replace("image_2", "image_3")
            imgA, imgB, size = utils.get_stereo_img(imgA_full_path, imgB_full_path)
            disparity_full_path = imgA_full_path.replace("image_2", "disp_noc_0")
            disp = utils.get_disparity(disparity_full_path, size)

            imgA = imgA.unsqueeze(0)
            imgB = imgB.unsqueeze(0)
            disp = disp.unsqueeze(0)
            if img0 is None:
                img0 = imgA
                img1 = imgB
                disparity = disp
            else:
                img0 = torch.cat((img0, imgA), axis=0)
                img1 = torch.cat((img1, imgB), axis=0)
                disparity = torch.cat((disparity, disp), axis=0)

        if img0.shape[0] < args.batch_size:
            continue
        else:
            print(f"{batch_idx = }")
            if args.mono:
                output_dir = f"output/monocular/{batch_idx // every_batch_change_dir}"
            else:
                output_dir = f"output/stereo/{batch_idx // every_batch_change_dir}"
            if batch_idx % every_batch_change_dir == 0:
                os.mkdir(output_dir)
                print(f"change dir to {batch_idx // every_batch_change_dir}")

            cur_batch_start_time = time.time()
            with torch.no_grad():
                if args.mono:
                    img0 = img0.to(device)
                    img0_depth = img0_depth.to(device)
                    preprocess_mono(batch_idx, img0, img0_depth, output_dir)
                else:
                    img0 = img0.to(device)
                    img1 = img1.to(device)
                    disparity = disparity.to(device)
                    preprocess_stereo(batch_idx, img0, img1, disparity, output_dir)
            cur_batch_end_time = time.time()
            print(f"cur batch spend time = {cur_batch_end_time - cur_batch_start_time}")
        
            batch_idx = batch_idx + 1
            img0 = None
            img0_depth = None
            torch.cuda.empty_cache()
            gc.collect()
            # if batch_idx == 1250:
            #     break

        # if cur_batch_end_time - cur_batch_start_time > 30:
        #     print(f"{batch_idx = }")
        #     break

        # break

    torch.cuda.empty_cache()

    end = time.time()
    print(f"total cal time = {end - start}")
