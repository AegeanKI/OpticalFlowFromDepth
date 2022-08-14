import utils
# import using_clib
from using_clib import Resample2d, ForwardWarping
import geometry  
import flow_colors
import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import ctypes
import matplotlib.pyplot as plt
from bilateral_filter import sparse_bilateral_filtering
import argparse
import time
import skimage

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
                       [0,       0,   0, 1]]], dtype=np.float32)

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

        axisangle = torch.from_numpy(np.array([[camera_ang]], dtype=np.float32))
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
        print("converting depth to disparity")
        h, w = size
        s = utils.get_random(175, 50, random_sign=False)
        
        depth_max = np.max(depth)
        disparity = (1. / depth) * s * depth_max / w
        # B, f = Plausible.B(), Plausible.f()
        # disparity = B * f / depth
        return disparity

    @staticmethod
    def disparity_to_flow(disparity, size, random_sign=True):
        print("converting disparity to flow")
        h, w = size
        flow = np.zeros((1, h, w, 2))
        flow[0, :, :, 0] = -disparity
        if random_sign:
            flow = flow * utils.get_random(0, 1)
        flow = torch.from_numpy(flow).double()
        return flow

    @staticmethod
    def disparity_to_depth(disparity):
        print("converting disparity to depth")
        B, f = Plausible.B(), Plausible.f()
        depth = B * f / (disparity + 0.25)
        depth[depth > 100] = 100
        return depth

    @staticmethod
    def depth_to_random_flow(depth, size, segment=None):
        print("converting depth to random flow")
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
        backward_flow = ForwardWarping()(flow_np, flow, depth, size)
        backward_flow = torch.from_numpy(backward_flow).unsqueeze(0) * -1.
        return backward_flow

    @staticmethod
    def two_contiguous_flows_to_one_flow(flow1, backward_flow1, flow2, mid_depth, size):
        flow2_np = flow2[0].detach().numpy()
        concat_flow = ForwardWarping()(flow2_np, backward_flow1, mid_depth, size)
        concat_flow = torch.from_numpy(concat_flow).unsqueeze(0)
        concat_flow = concat_flow + flow1
        return concat_flow

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono", type=int, default=0, help="use monocular or not")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    start = time.time()

    img0, img0_depth, img0_mask, size = None, None, None, None
    output_dir = None
    if args.mono:
        output_dir = "output/monocular"
        img0, size = utils.get_img("../img.jpg")
        img0_depth  = utils.get_depth("../depth.png", size, 16) # depth.png: 0 ~ 65535
        img0_depth = utils.fix_depth(img0_depth, img0)
        disparity = Convert.depth_to_disparity(img0_depth, size)
        flow01 = Convert.disparity_to_flow(disparity, size)
    else:
        output_dir = "output/stereo"
        img0, img1_real, size = utils.get_stereo_img("../left.png", "../right.png")
        disparity = utils.get_disparity("../disparity.png", size)
        img0_depth = Convert.disparity_to_depth(disparity)
        img0_depth = utils.fix_depth(img0_depth, img0)
        flow01 = Convert.disparity_to_flow(disparity, size, False)

    cv2.imwrite(f"{output_dir}/img0.png", img0)
    plt.imsave(f"{output_dir}/img0_depth.png", 1 / img0_depth, cmap="magma")
    np.save(f"{output_dir}/img0.npy", img0)
    np.save(f"{output_dir}/img0_depth.npy", img0_depth)

    back_flow01 = Convert.flow_to_backward_flow(flow01, img0_depth, size)
    _, flow01_color = utils.color_flow(flow01)
    _, back_flow01_color = utils.color_flow(back_flow01)
    cv2.imwrite(f"{output_dir}/flow01.png", flow01_color)
    cv2.imwrite(f"{output_dir}/back_flow01.png", back_flow01_color)
    torch.save(flow01, f"{output_dir}/flow01.pt")
    torch.save(back_flow01, f"{output_dir}/back_flow01.pt")

    img1 = ForwardWarping()(img0, flow01, img0_depth, size)
    img1_depth = ForwardWarping()(img0_depth, flow01, img0_depth, size)
    img1_depth = utils.fix_depth(img1_depth, img1)
    cv2.imwrite(f"{output_dir}/img1.png", img1)
    plt.imsave(f"{output_dir}/img1_depth.png", 1 / img1_depth, cmap="magma")
    np.save(f"{output_dir}/img1.npy", img1)
    np.save(f"{output_dir}/img1_depth.npy", img1_depth)
    
    flow12 = Convert.depth_to_random_flow(img1_depth, size)
    back_flow12 = Convert.flow_to_backward_flow(flow12, img1_depth, size)
    _, flow12_color = utils.color_flow(flow12)
    _, back_flow12_color = utils.color_flow(back_flow12)
    cv2.imwrite(f"{output_dir}/flow12.png", flow12_color)
    cv2.imwrite(f"{output_dir}/back_flow12.png", back_flow12_color)
    torch.save(flow12, f"{output_dir}/flow12.pt")
    torch.save(back_flow12, f"{output_dir}/back_flow12.pt")

    img2 = ForwardWarping()(img1, flow12, img1_depth, size)
    img2_depth = ForwardWarping()(img1_depth, flow12, img1_depth, size)
    img2_depth = utils.fix_depth(img2_depth, img1)
    cv2.imwrite(f"{output_dir}/img2.png", img2)
    plt.imsave(f"{output_dir}/img2_depth.png", 1 / img2_depth, cmap="magma")
    np.save(f"{output_dir}/img2.npy", img2)
    np.save(f"{output_dir}/img2_depth.npy", img2_depth)
    
    flow02 = Convert.two_contiguous_flows_to_one_flow(flow01, back_flow01, flow12,
                                                      img1_depth, size)
    back_flow02 = Convert.flow_to_backward_flow(flow02, img0_depth, size)
    _, flow02_color = utils.color_flow(flow02)
    _, back_flow02_color = utils.color_flow(back_flow02)
    cv2.imwrite(f"{output_dir}/flow02.png", flow02_color)
    cv2.imwrite(f"{output_dir}/back_flow02.png", back_flow02_color)
    torch.save(flow02, f"{output_dir}/flow02.pt")
    torch.save(back_flow02, f"{output_dir}/back_flow02.pt")

    test02 = ForwardWarping()(img0, flow02, img0_depth, size)
    cv2.imwrite(f"{output_dir}/test02.png", test02)

    end = time.time()
    print(f"cal time = {end - start}")
    torch.cuda.empty_cache()
