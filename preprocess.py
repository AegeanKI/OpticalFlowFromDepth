import utils
# import using_clib
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
        h, w = size
        s = utils.get_random(175, 50, random_sign=False)
        
        depth_max = np.max(depth)
        disparity = (1. / depth) * s * depth_max / w
        return disparity

    @staticmethod
    def disparity_to_flow(disparity, size, random_sign=True):
        h, w = size
        flow = np.zeros((1, h, w, 2))
        flow[0, :, :, 0] = -disparity
        if random_sign:
            flow = flow * utils.get_random(0, 1)
        flow = torch.from_numpy(flow).float()
        return flow

    @staticmethod
    def disparity_to_depth(disparity):
        B, f = Plausible.B(), Plausible.f()
        depth = B * f / (disparity + 0.005)
        depth = (np.max(depth) - depth) / (np.max(depth) - np.min(depth))
        depth = utils.normalize_depth(depth)
        return depth, B

    @staticmethod
    def depth_to_random_flow(depth, size, segment=None):
        h, w = size
        K, inv_K = Plausible.K(size)
        backproject_depth = geometry.BackprojectDepth(1, h, w)
        project_3d = geometry.Project3D(1, h, w)

        depth = torch.from_numpy(depth).unsqueeze(0).float()
        cam_points = backproject_depth(depth, inv_K)

        T1, axisangle, translation = Plausible.random_motion(1. / 72., 1. / 72., 0.05, 0.05) 
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


def flow_forward_warping(img, flow, depth, size):
    # print(f"{img.shape = }") (h, w) or (h, w, c)
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

    # res = warped_img[0, :, :, 0] if one_channel else warped_img[0, :, :, 0:3]
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
    safe_y = np.maximum(np.minimum(p1[:, :, :, 1].long(), h - 1), 0)
    safe_x = np.maximum(np.minimum(p1[:, :, :, 0].long(), w - 1), 0)
    img = img.reshape((1, h, w, 3))
    warped = np.zeros((h, w, 5))
    dlut = np.ones(size) * 1000
    for i in range(h):
        for j in range(w):
            x = safe_x[0, i, j]
            y = safe_y[0, i, j]
            if depth[0, i, j] < dlut[y, x]:
                for c in range(3):
                    warped[y, x, c] = img[0, i, j, c]

            warped[y, x, 3] = 1 
            if dlut[y, x] != 1000:
                warped[y, x, 4] = 0
            else:
                warped[y, x, 4] = 1
            dlut[y, x] = depth[0, i, j]

    warped = warped.reshape(1, h, w, 5)
    return warped

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono", type=int, default=0, help="use monocular or not")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.mono:
        output_dir = "output/monocular"
        img, size = utils.get_img("../im0.jpg") # CHW
        depth  = utils.get_depth("../d0.png", size, 16) # HW
        depth = sparse_bilateral_filtering(depth.copy(), img.copy(),
                                           filter_size=[5, 5], num_iter=2)
        cv2.imwrite(f"{output_dir}/original_img.png", img)
        plt.imsave(f"{output_dir}/original_depth.png", 1 / depth, cmap="magma")

        disparity = Convert.depth_to_disparity(depth, size)
        plt.imsave(f"{output_dir}/depth_to_disparity.png", disparity, cmap="magma")

        stereo_flow = Convert.disparity_to_flow(disparity, size)
        _, stereo_flow_color = utils.color_flow(stereo_flow)
        cv2.imwrite(f"{output_dir}/stereo_flow_color.png", stereo_flow_color)

        stereo_img, stereo_masks = flow_forward_warping(img, stereo_flow, depth, size)
        stereo_depth, _ = flow_forward_warping(depth, stereo_flow, depth, size)
        stereo_depth[stereo_depth == 0] = 100
        # stereo_depth = sparse_bilateral_filtering(stereo_depth.copy(), stereo_img.copy(),
        #                                           filter_size=[5, 5], num_iter=2)
        cv2.imwrite(f"{output_dir}/stereo_img.png", stereo_img)
        plt.imsave(f"{output_dir}/stereo_depth.png", 1 / stereo_depth, cmap="magma")
        cv2.imwrite(f"{output_dir}/stereo_masks_H.png", stereo_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/stereo_masks_M.png", stereo_masks["M"] * 255)
        # cv2.imwrite(f"{output_dir}/stereo_masks_P.png", stereo_masks["P"] * 255)

        # =====
        stereo_flow_np = stereo_flow[0].detach().numpy()
        backward_stereo_flow, backward_stereo_masks = flow_forward_warping(stereo_flow_np, stereo_flow, depth, size)
        backward_stereo_flow = torch.from_numpy(backward_stereo_flow).unsqueeze(0) * -1
        _, backward_stereo_flow_color = utils.color_flow(backward_stereo_flow)
        cv2.imwrite(f"{output_dir}/backward_stereo_flow_color.png", backward_stereo_flow_color)
        backward_stereo_img, _ = flow_forward_warping(stereo_img, backward_stereo_flow, stereo_depth, size)
        cv2.imwrite(f"{output_dir}/backward_stereo_img.png", backward_stereo_img)
        cv2.imwrite(f"{output_dir}/backward_stereo_H.png", backward_stereo_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/backward_stereo_M.png", backward_stereo_masks["M"] * 255)
        # cv2.imwrite(f"{output_dir}/backward_stereo_P.png", backward_stereo_masks["P"] * 255)
        
        # quit()
        # =====

        moving_flow = Convert.depth_to_random_flow(stereo_depth, size)
        _, moving_flow_color = utils.color_flow(moving_flow)
        cv2.imwrite(f"{output_dir}/moving_flow_color.png", moving_flow_color)

        moving_img, moving_masks = flow_forward_warping(stereo_img, moving_flow, stereo_depth, size)
        moving_depth, _ = flow_forward_warping(stereo_depth, moving_flow, stereo_depth, size)
        moving_depth[moving_depth == 0] = 100
        # moving_depth = sparse_bilateral_filtering(moving_depth.copy(), moving_img.copy(),
        #                                           filter_size=[5, 5], num_iter=2)
        cv2.imwrite(f"{output_dir}/moving_img.png", moving_img)
        plt.imsave(f"{output_dir}/moving_depth.png", 1 / moving_depth, cmap="magma")
        cv2.imwrite(f"{output_dir}/moving_masks_H.png", moving_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/moving_masks_M.png", moving_masks["M"] * 255)
        # cv2.imwrite(f"{output_dir}/moving_masks_P.png", moving_masks["P"] * 255)

        # =====
        moving_flow_np = moving_flow[0].detach().numpy()
        # backward_moving_flow, backward_moving_masks = flow_forward_warping(moving_flow_np, moving_flow, depth, size)
        backward_moving_flow, backward_moving_masks = flow_forward_warping(moving_flow_np, moving_flow, stereo_depth, size)
        backward_moving_flow = torch.from_numpy(backward_moving_flow).unsqueeze(0) * -1
        
        _, backward_moving_flow_color = utils.color_flow(backward_moving_flow)
        cv2.imwrite(f"{output_dir}/backward_moving_flow_color.png", backward_moving_flow_color)
        backward_moving_img, _ = flow_forward_warping(moving_img, backward_moving_flow, moving_depth, size)
        cv2.imwrite(f"{output_dir}/backward_moving_img.png", backward_moving_img)
        cv2.imwrite(f"{output_dir}/backward_moving_H.png", backward_moving_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/backward_moving_M.png", backward_moving_masks["M"] * 255)
        # cv2.imwrite(f"{output_dir}/backward_moving_P.png", backward_moving_masks["P"] * 255)
        # =====

        # concat_flow = moving_flow + stereo_flow
        moving_flow_np = moving_flow[0].detach().numpy()
        concat_flow, concat_masks = flow_forward_warping(moving_flow_np, backward_stereo_flow, stereo_depth, size)
        concat_flow = torch.from_numpy(concat_flow).unsqueeze(0)
        _, concat_flow_color = utils.color_flow(concat_flow)
        cv2.imwrite(f"{output_dir}/concat_flow_color_minus_stereo_flow.png", concat_flow_color)
        concat_flow = concat_flow + stereo_flow
        _, concat_flow_color = utils.color_flow(concat_flow)
        cv2.imwrite(f"{output_dir}/concat_flow_color.png", concat_flow_color)
        concat_img, concat_masks = flow_forward_warping(img, concat_flow, depth, size)
        m = concat_masks["H"]
        m = np.stack((m, m, m), -1)
        cv2.imwrite(f"{output_dir}/concat_img.png", concat_img * m)
        cv2.imwrite(f"{output_dir}/concat_masks_H.png", concat_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/concat_masks_M.png", concat_masks["M"] * 255)
        # cv2.imwrite(f"{output_dir}/concat_masks_P.png", concat_masks["P"] * 255)

    else:
        output_dir = "output/stereo"
        left, right, size = utils.get_stereo_img("../left.png", "../right.png")
        disparity = utils.get_disparity("../disparity-1.png", size)

        cv2.imwrite(f"{output_dir}/original_left.png", left)
        cv2.imwrite(f"{output_dir}/original_right.png", right)
        plt.imsave(f"{output_dir}/original_disparity.png", disparity, cmap="magma")

        depth, baseline = Convert.disparity_to_depth(disparity)
        depth = sparse_bilateral_filtering(depth.copy(), left.copy(),
                                           filter_size=[5, 5], num_iter=2)
        plt.imsave(f"{output_dir}/disparity_to_depth.png", 1 / depth, cmap="magma")

        stereo_flow = Convert.disparity_to_flow(disparity, size, False)
        _, stereo_flow_color = utils.color_flow(stereo_flow)
        cv2.imwrite(f"{output_dir}/stereo_flow_color.png", stereo_flow_color)
        stereo_img, stereo_masks = flow_forward_warping(left, stereo_flow, depth, size)
        stereo_depth, _ = flow_forward_warping(depth, stereo_flow, depth, size)
        stereo_depth[stereo_depth == 0] = 100

        cv2.imwrite(f"{output_dir}/stereo_img.png", stereo_img)
        plt.imsave(f"{output_dir}/stereo_depth.png", 1 / stereo_depth, cmap="magma")
        cv2.imwrite(f"{output_dir}/stereo_H.png", stereo_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/stereo_M.png", stereo_masks["M"] * 255)
        cv2.imwrite(f"{output_dir}/stereo_P.png", stereo_masks["P"] * 255)

        # =====
        stereo_flow_np = stereo_flow[0].detach().numpy()
        backward_stereo_flow, backward_stereo_masks = flow_forward_warping(stereo_flow_np, stereo_flow, depth, size)
        backward_stereo_flow = torch.from_numpy(backward_stereo_flow).unsqueeze(0) * -1
        _, backward_stereo_flow_color = utils.color_flow(backward_stereo_flow)
        cv2.imwrite(f"{output_dir}/backward_stereo_flow_color.png", backward_stereo_flow_color)
        backward_stereo_img, _ = flow_forward_warping(stereo_img, backward_stereo_flow, stereo_depth, size)
        cv2.imwrite(f"{output_dir}/backward_stereo_img.png", backward_stereo_img)
        cv2.imwrite(f"{output_dir}/backward_stereo_H.png", backward_stereo_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/backward_stereo_M.png", backward_stereo_masks["M"] * 255)
        # cv2.imwrite(f"{output_dir}/backward_stereo_P.png", backward_stereo_masks["P"] * 255)
        
        # quit()
        # =====

        moving_flow = Convert.depth_to_random_flow(stereo_depth, size)
        _, moving_flow_color = utils.color_flow(moving_flow)
        cv2.imwrite(f"{output_dir}/moving_flow_color.png", moving_flow_color)

        moving_img, _ = flow_forward_warping(stereo_img, moving_flow, stereo_depth, size)
        moving_depth, _ = flow_forward_warping(stereo_depth, moving_flow, stereo_depth, size)
        moving_depth[moving_depth == 0] = 100

        moving_masks = {}
        moving_masks["H"], _ = flow_forward_warping(stereo_masks["H"], moving_flow, stereo_depth, size)
        moving_masks["M"], _ = flow_forward_warping(stereo_masks["M"], moving_flow, stereo_depth, size)
        moving_masks["P"], _ = flow_forward_warping(stereo_masks["P"], moving_flow, stereo_depth, size)
        cv2.imwrite(f"{output_dir}/moving_img.png", moving_img)
        plt.imsave(f"{output_dir}/moving_depth.png", 1 / moving_depth, cmap="magma")
        cv2.imwrite(f"{output_dir}/moving_H.png", moving_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/moving_M.png", moving_masks["M"] * 255)

        # =====
        moving_flow_np = moving_flow[0].detach().numpy()
        # backward_moving_flow, backward_moving_masks = flow_forward_warping(moving_flow_np, moving_flow, depth, size)
        backward_moving_flow, backward_moving_masks = flow_forward_warping(moving_flow_np, moving_flow, stereo_depth, size)
        backward_moving_flow = torch.from_numpy(backward_moving_flow).unsqueeze(0) * -1
        
        _, backward_moving_flow_color = utils.color_flow(backward_moving_flow)
        cv2.imwrite(f"{output_dir}/backward_moving_flow_color.png", backward_moving_flow_color)
        backward_moving_img, _ = flow_forward_warping(moving_img, backward_moving_flow, moving_depth, size)
        cv2.imwrite(f"{output_dir}/backward_moving_img.png", backward_moving_img)
        cv2.imwrite(f"{output_dir}/backward_moving_H.png", backward_moving_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/backward_moving_M.png", backward_moving_masks["M"] * 255)
        # cv2.imwrite(f"{output_dir}/backward_moving_P.png", backward_moving_masks["P"] * 255)
        # =====

        # concat_flow = moving_flow + stereo_flow
        moving_flow_np = moving_flow[0].detach().numpy()
        concat_flow, concat_masks = flow_forward_warping(moving_flow_np, backward_stereo_flow, stereo_depth, size)
        concat_flow = torch.from_numpy(concat_flow).unsqueeze(0)
        _, concat_flow_color = utils.color_flow(concat_flow)
        cv2.imwrite(f"{output_dir}/concat_flow_color_minus_stereo_flow.png", concat_flow_color)
        concat_flow = concat_flow + stereo_flow
        _, concat_flow_color = utils.color_flow(concat_flow)
        cv2.imwrite(f"{output_dir}/concat_flow_color.png", concat_flow_color)
        concat_img, concat_masks = flow_forward_warping(left, concat_flow, depth, size)
        m = concat_masks["H"]
        m = np.stack((m, m, m), -1)
        cv2.imwrite(f"{output_dir}/concat_img.png", concat_img * m)
        cv2.imwrite(f"{output_dir}/concat_masks_H.png", concat_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/concat_masks_M.png", concat_masks["M"] * 255)
        # cv2.imwrite(f"{output_dir}/concat_masks_P.png", concat_masks["P"] * 255)

