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
    if args.mono:
        output_dir = "output/monocular"
        img, size = utils.get_img("../img.jpg")
        depth  = utils.get_depth("../depth.png", size, 16) # depth.png: 0 ~ 65535

        depth = sparse_bilateral_filtering(depth.copy(), img.copy(),
                                           filter_size=[5, 5], num_iter=2)

        cv2.imwrite(f"{output_dir}/original_img.png", img)
        plt.imsave(f"{output_dir}/original_depth.png", 1 / depth, cmap="magma")

        disparity = Convert.depth_to_disparity(depth, size)
        plt.imsave(f"{output_dir}/depth_to_disparity.png", disparity, cmap="magma")

        stereo_flow = Convert.disparity_to_flow(disparity, size)
        _, stereo_flow_color = utils.color_flow(stereo_flow)
        cv2.imwrite(f"{output_dir}/stereo_flow_color.png", stereo_flow_color)

        print("warping stereo image")
        stereo_img, stereo_masks = flow_forward_warping(img, stereo_flow, depth, size)
        stereo_depth, _ = flow_forward_warping(depth, stereo_flow, depth, size)
        stereo_depth[stereo_depth == 0] = 100
        stereo_depth = sparse_bilateral_filtering(stereo_depth.copy(), stereo_img.copy(),
                                                  filter_size=[5, 5], num_iter=2)
        cv2.imwrite(f"{output_dir}/stereo_img.png", stereo_img)
        plt.imsave(f"{output_dir}/stereo_depth.png", 1 / stereo_depth, cmap="magma")
        cv2.imwrite(f"{output_dir}/stereo_masks_H.png", stereo_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/stereo_masks_M.png", stereo_masks["M"] * 255)
        # cv2.imwrite(f"{output_dir}/stereo_masks_P.png", stereo_masks["P"] * 255)
        # cv2.imwrite(f"{output_dir}/stereo_masks_H'.png", stereo_masks["H'"] * 255)
        # cv2.imwrite(f"{output_dir}/stereo_masks_M'.png", stereo_masks["M'"] * 255)

        print("warping backward stereo flow")
        backward_stereo_flow, backward_stereo_masks = Convert.flow_to_backward_flow(stereo_flow, depth, size)
        _, backward_stereo_flow_color = utils.color_flow(backward_stereo_flow)
        cv2.imwrite(f"{output_dir}/backward_stereo_flow_color.png", backward_stereo_flow_color)

        print("warping backward stereo image")
        backward_stereo_img, _ = flow_forward_warping(stereo_img, backward_stereo_flow, stereo_depth, size)
        # backward_stereo_masks_H, _ = flow_forward_warping(stereo_masks["H"], backward_stereo_flow, stereo_depth, size)
        # backward_stereo_masks_M, _ = flow_forward_warping(stereo_masks["M"], backward_stereo_flow, depth, size)

        # h, w = size
        # for i in range(h):
        #     for j in range(w):
        #         if not backward_stereo_masks_H[i, j] and not backward_stereo_masks["H"][i, j]:
        #             backward_stereo_masks_H[i, j] = 1

        #         backward_stereo_masks["M"][i, j] = 1
        #         if stereo_masks["H"][i, j]:
        #             backward_stereo_masks["M"][i, j] = 0
        # backward_stereo_masks["H"] = backward_stereo_masks_H 
        cv2.imwrite(f"{output_dir}/backward_stereo_img.png", backward_stereo_img)
        cv2.imwrite(f"{output_dir}/backward_stereo_masks_H.png", backward_stereo_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/backward_stereo_masks_M.png", backward_stereo_masks["M"] * 255)
        # cv2.imwrite(f"{output_dir}/backward_stereo_masks_H_test.png", backward_stereo_masks_H * 255)
        # cv2.imwrite(f"{output_dir}/backward_stereo_masks_M_test.png", backward_stereo_masks_M * 255)

        moving_flow = Convert.depth_to_random_flow(stereo_depth, size)
        _, moving_flow_color = utils.color_flow(moving_flow)
        cv2.imwrite(f"{output_dir}/moving_flow_color.png", moving_flow_color)

        print("warping moving image")
        moving_img, moving_masks = flow_forward_warping(stereo_img, moving_flow, stereo_depth, size)
        moving_depth, _ = flow_forward_warping(stereo_depth, moving_flow, stereo_depth, size)
        moving_depth[moving_depth == 0] = 100
        moving_depth = sparse_bilateral_filtering(moving_depth.copy(), moving_img.copy(),
                                                  filter_size=[5, 5], num_iter=2)
        moving_masks_H, _ = flow_forward_warping(stereo_masks["H"], moving_flow, stereo_depth, size)
        moving_masks_M, _ = flow_forward_warping(stereo_masks["M"], moving_flow, stereo_depth, size)
        moving_masks["H"] = moving_masks["H"] * moving_masks_H
        moving_masks["M"] = moving_masks["M"] * moving_masks_M
        cv2.imwrite(f"{output_dir}/moving_img.png", moving_img)
        plt.imsave(f"{output_dir}/moving_depth.png", 1 / moving_depth, cmap="magma")
        cv2.imwrite(f"{output_dir}/moving_masks_H.png", moving_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/moving_masks_M.png", moving_masks["M"] * 255)

        print("warping backward moving flow")
        backward_moving_flow, backward_moving_masks = Convert.flow_to_backward_flow(moving_flow, stereo_depth, size)

        print("warping backward moving image")
        _, backward_moving_flow_color = utils.color_flow(backward_moving_flow)
        cv2.imwrite(f"{output_dir}/backward_moving_flow_color.png", backward_moving_flow_color)
        backward_moving_img, _ = flow_forward_warping(moving_img, backward_moving_flow, moving_depth, size)
        cv2.imwrite(f"{output_dir}/backward_moving_img.png", backward_moving_img)
        cv2.imwrite(f"{output_dir}/backward_moving_masks_H.png", backward_moving_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/backward_moving_masks_M.png", backward_moving_masks["M"] * 255)

        print("warping concat image")
        concat_flow = Convert.two_contiguous_flows_to_one_flow(stereo_flow, backward_stereo_flow,
                                                      moving_flow, stereo_depth, size)
        _, concat_flow_color = utils.color_flow(concat_flow)
        cv2.imwrite(f"{output_dir}/concat_flow_color.png", concat_flow_color)
        concat_img, concat_masks = flow_forward_warping(img, concat_flow, depth, size)
        m = moving_masks["H"]
        m = np.stack((m, m, m), -1)
        cv2.imwrite(f"{output_dir}/concat_img.png", concat_img * m)
        cv2.imwrite(f"{output_dir}/concat_masks_H.png", concat_masks["H"] * moving_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/concat_masks_M.png", concat_masks["M"] * moving_masks["H"] * 255)
    else:
        output_dir = "output/stereo"
        left, right, size = utils.get_stereo_img("../left.png", "../right.png")
        disparity = utils.get_disparity("../disparity.png", size)
        valid = (disparity > 0)

        cv2.imwrite(f"{output_dir}/original_left.png", left)
        cv2.imwrite(f"{output_dir}/original_right.png", right)
        plt.imsave(f"{output_dir}/original_disparity.png", disparity, cmap="magma")
        cv2.imwrite(f"{output_dir}/original_valid.png", valid * 255)

        depth = Convert.disparity_to_depth(disparity)
        depth = sparse_bilateral_filtering(depth.copy(), left.copy(),
                                           filter_size=[5, 5], num_iter=2)
        plt.imsave(f"{output_dir}/disparity_to_depth.png", 1 / depth, cmap="magma")

        stereo_flow = Convert.disparity_to_flow(disparity, size, False)
        _, stereo_flow_color = utils.color_flow(stereo_flow)
        cv2.imwrite(f"{output_dir}/stereo_flow_color.png", stereo_flow_color)

        print("warping stereo image")
        stereo_img, stereo_masks = flow_forward_warping(left, stereo_flow, depth, size)
        stereo_depth, _ = flow_forward_warping(depth, stereo_flow, depth, size)
        stereo_valid, _ = flow_forward_warping(valid, stereo_flow, depth, size)

        stereo_depth[stereo_depth == 0] = 100
        stereo_depth = sparse_bilateral_filtering(stereo_depth.copy(), stereo_img.copy(),
                                                  filter_size=[5, 5], num_iter=2)

        stereo_valid_3c = np.stack((stereo_valid, stereo_valid, stereo_valid), -1)

        cv2.imwrite(f"{output_dir}/stereo_img.png", stereo_img * stereo_valid_3c)
        plt.imsave(f"{output_dir}/stereo_depth.png", 1 / stereo_depth, cmap="magma")
        cv2.imwrite(f"{output_dir}/stereo_masks_H.png", stereo_masks["H"] * stereo_valid * 255)
        cv2.imwrite(f"{output_dir}/stereo_masks_M.png", stereo_masks["M"] * stereo_valid * 255)

        print("warping backward stereo flow")

        backward_stereo_flow, backward_stereo_masks = Convert.flow_to_backward_flow(stereo_flow, depth, size)
        _, backward_stereo_flow_color = utils.color_flow(backward_stereo_flow)
        cv2.imwrite(f"{output_dir}/backward_stereo_flow_color.png", backward_stereo_flow_color)

        print("warping backward stereo image")
        backward_stereo_img, _ = flow_forward_warping(stereo_img, backward_stereo_flow, stereo_depth, size)
        cv2.imwrite(f"{output_dir}/backward_stereo_img.png", backward_stereo_img)
        cv2.imwrite(f"{output_dir}/backward_stereo_masks_H.png", backward_stereo_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/backward_stereo_masks_M.png", backward_stereo_masks["M"] * 255)

        moving_flow = Convert.depth_to_random_flow(stereo_depth, size)
        _, moving_flow_color = utils.color_flow(moving_flow)
        cv2.imwrite(f"{output_dir}/moving_flow_color.png", moving_flow_color)

        print("warping moving image")
        moving_img, moving_masks = flow_forward_warping(stereo_img, moving_flow, stereo_depth, size)
        moving_depth, _ = flow_forward_warping(stereo_depth, moving_flow, stereo_depth, size)
        moving_depth[moving_depth == 0] = 100
        moving_depth = sparse_bilateral_filtering(moving_depth.copy(), moving_img.copy(),
                                                  filter_size=[5, 5], num_iter=2)
        moving_masks_H, _ = flow_forward_warping(stereo_masks["H"], moving_flow, stereo_depth, size)
        moving_masks_M, _ = flow_forward_warping(stereo_masks["M"], moving_flow, stereo_depth, size)
        moving_masks["H"] = moving_masks["H"] * moving_masks_H
        moving_masks["M"] = moving_masks["M"] * moving_masks_M
        moving_valid, _ = flow_forward_warping(stereo_valid, moving_flow, stereo_depth, size)
        moving_valid_3c = np.stack((moving_valid, moving_valid, moving_valid), -1)
        cv2.imwrite(f"{output_dir}/moving_img.png", moving_img * moving_valid_3c)
        plt.imsave(f"{output_dir}/moving_depth.png", 1 / moving_depth, cmap="magma")
        cv2.imwrite(f"{output_dir}/moving_masks_H.png", moving_masks["H"] * moving_valid * 255)
        cv2.imwrite(f"{output_dir}/moving_masks_M.png", moving_masks["M"] * moving_valid * 255)

        print("warping backward moving flow")
        backward_moving_flow, backward_moving_masks = Convert.flow_to_backward_flow(moving_flow, stereo_depth, size)

        print("warping backward moving image")
        _, backward_moving_flow_color = utils.color_flow(backward_moving_flow)
        cv2.imwrite(f"{output_dir}/backward_moving_flow_color.png", backward_moving_flow_color)
        backward_moving_img, _ = flow_forward_warping(moving_img, backward_moving_flow, moving_depth, size)
        cv2.imwrite(f"{output_dir}/backward_moving_img.png", backward_moving_img)
        cv2.imwrite(f"{output_dir}/backward_moving_masks_H.png", backward_moving_masks["H"] * 255)
        cv2.imwrite(f"{output_dir}/backward_moving_masks_M.png", backward_moving_masks["M"] * 255)

        print("warping concat image")

        concat_flow = Convert.two_contiguous_flows_to_one_flow(stereo_flow, backward_stereo_flow,
                                                      moving_flow, stereo_depth, size)

        _, concat_flow_color = utils.color_flow(concat_flow)
        cv2.imwrite(f"{output_dir}/concat_flow_color.png", concat_flow_color)
        concat_img, concat_masks = flow_forward_warping(left, concat_flow, depth, size)
        m = moving_masks["H"] * concat_masks["H"]
        m = np.stack((m, m, m), -1)
        cv2.imwrite(f"{output_dir}/concat_img.png", concat_img * m * moving_valid_3c)
        cv2.imwrite(f"{output_dir}/concat_masks_H.png", concat_masks["H"] * moving_valid * 255)
        cv2.imwrite(f"{output_dir}/concat_masks_M.png", concat_masks["M"] * moving_valid * 255)

    end = time.time()
    print(f"cal time = {end - start}")

    torch.cuda.empty_cache()
