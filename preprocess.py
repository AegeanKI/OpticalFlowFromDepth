import utils
# import using_clib
from using_clib import Resample2d
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
        flow = torch.from_numpy(flow).float()
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
        
        meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
        p0 = np.stack(meshgrid, axis=-1).astype(np.float32)   
        flow = p1 - p0
        return flow

    @staticmethod
    def flow_to_backward_flow(flow, depth, size):
        flow_np = flow[0].detach().numpy()
        backward_flow, backward_masks = flow_forward_warping(flow_np, flow, depth, size)
        backward_flow = torch.from_numpy(backward_flow).unsqueeze(0) * -1.
        return backward_flow, backward_masks

    @staticmethod
    def two_contiguous_flows_to_one_flow(flow1, backward_flow1, flow2, mid_depth, size):
        flow2_np = flow2[0].detach().numpy()
        concat_flow, concat_masks = flow_forward_warping(flow2_np, backward_flow1, mid_depth, size)
        concat_flow = torch.from_numpy(concat_flow).unsqueeze(0)
        concat_flow = concat_flow + flow1
        return concat_flow


def flow_forward_warping(img, flow, depth, size):
    h, w = size
    one_channel = (img.ndim == 2)
    if one_channel:
        img = np.stack((img, img, img), -1) 
    
    two_channel = (img.shape[2] == 2)
    if two_channel:
        z = np.zeros((h, w, 3))
        z[..., 0:2] = img
        img = z
    meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
    p0 = np.stack(meshgrid, axis=-1).astype(np.float32)
    p0 = torch.from_numpy(p0).unsqueeze(0)

    depth = torch.from_numpy(depth).unsqueeze(0).float()
    p1 = p0 + flow

    # warped_img = using_clib.forward_warping(img, p1, depth, size) 
    warped_img = forward_warping(img, p1, depth, size) 

    res = warped_img[0, :, :, 0:3]
    if one_channel:
        res = res[..., 0]
    if two_channel:
        res = res[..., 0:2]

    masks = {}
    masks["H"] = warped_img[0 ,:, :, 3]
    masks["M"] = warped_img[0, :, :, 4]
    masks["M"] = 1 - (masks["M"] == masks["H"]).astype(np.uint8)
    masks["M'"] = cv2.dilate(masks["M"], np.ones((3, 3), np.uint8), iterations=1)
    masks["P"] = (masks["M'"] == masks["M"]).astype(np.uint8)
    masks["H'"] = masks["H"] * masks["P"]
    return res, masks

def forward_warping(img, p1, depth, size):
    # print(f"{img.shape = }")    (h, w, c)
    # print(f"{depth.shape = }")  (1, h, w)
    h, w = size
    safe_y = np.maximum(np.minimum(p1[:, :, :, 1], h - 1), 0)
    safe_x = np.maximum(np.minimum(p1[:, :, :, 0], w - 1), 0)
    img = img.reshape((1, h, w, 3))
    warped = np.zeros((h, w, 5))
    dlut = np.ones(size) * 1000
    for i in range(h):
        for j in range(w):
            x = safe_x[0, i, j]
            y = safe_y[0, i, j]


            if warped[int(y), int(x), 3] == 0 or depth[0, i, j] < dlut[int(y), int(x)] - 1.:
                pass
            elif ((x - int(x) > 0.5) and
                  (warped[int(y), int(x) + 1, 3] == 0 or depth[0, i, j] < dlut[int(y), int(x) + 1] - 1.)):
                x = x + 1
            elif ((y - int(y) > 0.5) and
                  (warped[int(y) + 1, int(x), 3] == 0 or depth[0, i, j] < dlut[int(y) + 1, int(x)] - 1.)):
                y = y + 1
            elif ((x - int(x) > 0.5) and (y - int(y) > 0.5) and
                  (warped[int(y) + 1, int(x) + 1, 3] == 0 or depth[0, i, j] < dlut[int(y) + 1, int(x) + 1] - 1.)):
                x = x + 1
                y = y + 1

            y = int(y)
            x = int(x)

            if depth[0, i, j] < dlut[y, x]:
                for c in range(3):
                    warped[y, x, c] = img[0, i, j, c]

            warped[y, x, 3] = 1 
            if dlut[y, x] != 1000:
                warped[y, x, 4] = 0
            else:
                warped[y, x, 4] = 1
            dlut[y, x] = depth[0, i, j]

    onepixel_blank = True
    while onepixel_blank:
        onepixel_blank = False
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if warped[i, j, 3] == 1:
                    continue
                if warped[i, j - 1, 3] == 1 and warped[i, j + 1, 3] == 1:
                    warped[i, j, 0:3] = (warped[i, j - 1, 0:3] + warped[i, j + 1, 0:3]) / 2
                    warped[i, j, 3] = 1
                    warped[i, j, 4] = 0
                    onepixel_blank = True
                elif warped[i - 1, j, 3] == 1 and warped[i + 1, j, 3] == 1:
                    warped[i, j, 0:3] = (warped[i - 1, j, 0:3] + warped[i + 1, j, 0:3]) / 2
                    warped[i, j, 3] = 1
                    warped[i, j, 4] = 0
                    onepixel_blank = True
                elif warped[i - 1, j - 1, 3] == 1 and warped[i + 1, j + 1, 3] == 1:
                    warped[i, j, 0:3] = (warped[i - 1, j - 1, 0:3] + warped[i + 1, j + 1, 0:3]) / 2
                    warped[i, j, 3] = 1
                    warped[i, j, 4] = 0
                    onepixel_blank = True
                elif warped[i - 1, j + 1, 3] == 1 and warped[i + 1, j - 1, 3] == 1:
                    warped[i, j, 0:3] = (warped[i - 1, j + 1, 0:3] + warped[i + 1, j - 1, 0:3]) / 2
                    warped[i, j, 3] = 1
                    warped[i, j, 4] = 0
                    onepixel_blank = True
    warped = warped.reshape(1, h, w, 5)
    return warped

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
        img0_mask = np.ones(size)
        flow01 = Convert.disparity_to_flow(disparity, size)
    else:
        output_dir = "output/stereo"
        img0, img1_real, size = utils.get_stereo_img("../left.png", "../right.png")
        disparity = utils.get_disparity("../disparity.png", size)
        img0_depth = Convert.disparity_to_depth(disparity)
        img0_depth = utils.fix_depth(img0_depth, img0)
        img0_mask = (disparity > 0)
        flow01 = Convert.disparity_to_flow(disparity, size, False)

    cv2.imwrite(f"{output_dir}/img0.png", img0)
    plt.imsave(f"{output_dir}/img0_depth.png", 1 / img0_depth, cmap="magma")
    cv2.imwrite(f"{output_dir}/img0_mask.png", img0_mask * 255)
    np.save(f"{output_dir}/img0.npy", img0)
    np.save(f"{output_dir}/img0_depth.npy", img0_depth)
    np.save(f"{output_dir}/img0_mask.npy", img0_mask)
    # load_img0 = np.load(f"{output_dir}/img0.npy")
    # load_img0_depth = np.load(f"{output_dir}/img0_depth.npy")
    # load_img0_mask = np.load(f"{output_dir}/img0_mask.npy")


    back_flow01, _ = Convert.flow_to_backward_flow(flow01, img0_depth, size)
    _, flow01_color = utils.color_flow(flow01)
    _, back_flow01_color = utils.color_flow(back_flow01)
    cv2.imwrite(f"{output_dir}/flow01.png", flow01_color)
    cv2.imwrite(f"{output_dir}/back_flow01.png", back_flow01_color)
    torch.save(flow01, f"{output_dir}/flow01.pt")
    torch.save(back_flow01, f"{output_dir}/back_flow01.pt")
    # load_flow01 = torch.load(f"{output_dir}/flow01.pt")
    # load_back_flow01 = torch.load(f"{output_dir}/back_flow01.pt")

    img1, _ = flow_forward_warping(img0, flow01, img0_depth, size)
    img1_depth, _ = flow_forward_warping(img0_depth, flow01, img0_depth, size)
    img1_depth = utils.fix_depth(img1_depth, img1)
    img1_mask, _ = flow_forward_warping(img0_mask, flow01, img0_depth, size)
    cv2.imwrite(f"{output_dir}/img1.png", img1)
    plt.imsave(f"{output_dir}/img1_depth.png", 1 / img1_depth, cmap="magma")
    cv2.imwrite(f"{output_dir}/img1_mask.png", img1_mask * 255)
    np.save(f"{output_dir}/img1.npy", img1)
    np.save(f"{output_dir}/img1_depth.npy", img1_depth)
    np.save(f"{output_dir}/img1_mask.npy", img1_mask)


    flow12 = Convert.depth_to_random_flow(img1_depth, size)
    back_flow12, _ = Convert.flow_to_backward_flow(flow12, img1_depth, size)
    _, flow12_color = utils.color_flow(flow12)
    _, back_flow12_color = utils.color_flow(back_flow12)
    cv2.imwrite(f"{output_dir}/flow12.png", flow12_color)
    cv2.imwrite(f"{output_dir}/back_flow12.png", back_flow12_color)
    torch.save(flow01, f"{output_dir}/flow12.pt")
    torch.save(back_flow01, f"{output_dir}/back_flow12.pt")
    
    img2, _ = flow_forward_warping(img1, flow12, img1_depth, size)
    img2_depth, _ = flow_forward_warping(img1_depth, flow12, img1_depth, size)
    img2_depth = utils.fix_depth(img2_depth, img1)
    img2_mask, _ = flow_forward_warping(img1_mask, flow12, img1_depth, size)
    cv2.imwrite(f"{output_dir}/img2.png", img2)
    plt.imsave(f"{output_dir}/img2_depth.png", 1 / img2_depth, cmap="magma")
    cv2.imwrite(f"{output_dir}/img2_mask.png", img2_mask * 255)
    np.save(f"{output_dir}/img2.npy", img2)
    np.save(f"{output_dir}/img2_depth.npy", img2_depth)
    np.save(f"{output_dir}/img2_mask.npy", img2_mask)

    flow02 = Convert.two_contiguous_flows_to_one_flow(flow01, back_flow01, flow12,
                                                      img1_depth, size)
    back_flow02, _ = Convert.flow_to_backward_flow(flow02, img0_depth, size)
    _, flow02_color = utils.color_flow(flow02)
    _, back_flow02_color = utils.color_flow(back_flow02)
    cv2.imwrite(f"{output_dir}/flow02.png", flow02_color)
    cv2.imwrite(f"{output_dir}/back_flow02.png", back_flow02_color)
    torch.save(flow01, f"{output_dir}/flow02.pt")
    torch.save(back_flow01, f"{output_dir}/back_flow02.pt")

    end = time.time()
    print(f"cal time = {end - start}")
    torch.cuda.empty_cache()
