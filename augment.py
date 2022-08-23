import utils
import preprocess
import numpy as np
import math
import torch
# import torch.nn.functional as F
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import cv2
import matplotlib.pyplot as plt
import pykitti
from parser import DataParser

from preprocess import Convert
from bilateral_filter import sparse_bilateral_filtering
# from using_clib import Resample2d
from using_clib import ForwardWarping

class SpecialFlow():
    @staticmethod
    def flip(size):
        print("flip")
        h, w = size
        meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
        p0 = np.stack(meshgrid, axis=-1).astype(np.float32)

        horizontal = utils.get_random(0, 1) > 0
        if horizontal:
            meshgrid = np.meshgrid(range(w - 1, -1, -1), range(h), indexing="xy")
        else:
            meshgrid = np.meshgrid(range(w), range(h - 1, -1, -1), indexing="xy")
        p1 = np.stack(meshgrid, axis=-1).astype(np.float32)
        flip_flow = torch.from_numpy(p1 - p0).unsqueeze(0)
        backward_flip_flow = flip_flow
        return flip_flow, backward_flip_flow

    
    @staticmethod
    def rotate(size):
        print("rotate")
        h, w = size
        meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
        p0 = np.stack(meshgrid, axis=-1).astype(np.float32)

        c0 = np.random.randint(1, [3 * w, 3 * h]) - (w, h)
        theta = np.deg2rad(utils.get_random(10, 15))
        rotate = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        reverse_rotate = np.array([[np.cos(-theta), -np.sin(-theta)],
                                   [np.sin(-theta), np.cos(-theta)]])

        p1 = (p0 - c0) @ rotate + c0
        p_prev = (p0 - c0) @ reverse_rotate + c0
        rotate_flow = torch.from_numpy(p1 - p0).unsqueeze(0)
        back_rotate_flow = torch.from_numpy(p0 - p_prev).unsqueeze(0)
        return rotate_flow, back_rotate_flow

    @staticmethod
    def shear(size):
        print("shear")
        h, w = size
        meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
        p0 = np.stack(meshgrid, axis=-1).astype(np.float32)

        horizontal = utils.get_random(0, 1) > 0
        shear_range = utils.get_random(0.10, 0.15)
        if horizontal:
            shear = np.array([[1, 0], [shear_range, 1]])
            reverse_shear = np.array([[1, 0], [-shear_range, 1]])
        else:
            shear = np.array([[1, shear_range], [0, 1]])
            reverse_shear = np.array([[1, -shear_range], [0, 1]])
        
        p1 = p0 @ shear
        p_prev = p0 @ reverse_shear
        shear_flow = torch.from_numpy(p1 - p0).unsqueeze(0)
        back_shear_flow = torch.from_numpy(p0 - p_prev).unsqueeze(0)
        return shear_flow, back_shear_flow

def augment_flow(img0, img0_depth, img1, img1_depth,
                 flow01, back_flow01, size):
    augment_flow_type = utils.get_random(3, 0)
    augment_flow_type = 1.2
    if augment_flow_type >= 2.:
        special_flow_type = utils.get_random(3, 0, False)
        special_flow_type = 1.
        if special_flow_type >= 2.:
            special_flow, back_special_flow = SpecialFlow.shear(size)
        elif special_flow_type >= 1.:
            special_flow, back_special_flow = SpecialFlow.rotate(size)
        elif special_flow_type >= 0.:
            special_flow, back_special_flow = SpecialFlow.flip(size)

        augment_img0_flow = Convert.two_contiguous_flows_to_one_flow(special_flow, back_special_flow,
                                                                     flow01, img0_depth, size)
        augment_img1_flow = Convert.two_contiguous_flows_to_one_flow(flow01, back_flow01,
                                                                     special_flow, img1_depth, size)

        augment_img0 = ForwardWarping()(img0, special_flow, img0_depth, size)
        augment_img0_depth = ForwardWarping()(img0_depth, special_flow, img0_depth, size)
        augment_img0_depth = utils.fix_depth(augment_img0_depth, augment_img0)
        augment_img1 = ForwardWarping()(img1, special_flow, img1_depth, size)
        augment_img1_depth = ForwardWarping()(img1_depth, special_flow, img1_depth, size)
        augment_img1_depth = utils.fix_depth(augment_img1_depth, augment_img1)

        back_augment_img0_flow = Convert.flow_to_backward_flow(augment_img0_flow, augment_img0_depth, size)
        back_augment_img1_flow = Convert.flow_to_backward_flow(augment_img1_flow, img0_depth, size)

        return (augment_img0, augment_img0_depth,
                augment_img0_flow, back_augment_img0_flow, 
                img1, img1_depth), (
                img0, img0_depth,
                augment_img1_flow, back_augment_img1_flow,
                augment_img1, augment_img1_depth)
    elif augment_flow_type >= 1.:
        adjust_flow_type = utils.get_random(2, 0, False)
        adjust_flow_type = 1.2
        if adjust_flow_type >= 1.:
            h_len = int(utils.get_random(h / 4, h / 2, False))
            w_len = int(utils.get_random(w / 4, w / 2, False))

            h_start = int(utils.get_random(h - h_len, 0, False))
            w_start = int(utils.get_random(w - w_len, 0, False))

            img_mask = np.zeros((h, w, 3))
            img_mask[h_start:h_start + h_len, w_start:w_start + w_len] = 1
            depth_mask = np.zeros((h, w))
            depth_mask[h_start:h_start + h_len , w_start:w_start + w_len] = 1
            flow_mask = np.zeros((h, w, 2))
            flow_mask[h_start:h_start + h_len, w_start:w_start + w_len] = 1

            forward_img_mask = ForwardWarping()(img_mask, flow01, img0_depth, size)
            forward_depth_mask = ForwardWarping()(depth_mask, flow01, img0_depth, size)
            forward_flow_mask = ForwardWarping()(flow_mask, flow01, img0_depth, size)
            backward_img_mask = ForwardWarping()(img_mask, back_flow01, img1_depth, size)
            backward_depth_mask = ForwardWarping()(depth_mask, back_flow01, img1_depth, size)
            backward_flow_mask = ForwardWarping()(flow_mask, back_flow01, img1_depth, size)
        elif adjust_flow_type >= 0.:
            h_len = int(utils.get_random(h / 4, h / 4, False))
            w_len = int(utils.get_random(w / 4, w / 4, False))

            h_start = int(utils.get_random(h - h_len, 0, False))
            w_start = int(utils.get_random(w - w_len, 0, False))

            img_mask = np.ones((h, w, 3))
            img_mask[h_start:h_start + h_len, w_start:w_start + w_len] = 0
            depth_mask = np.ones((h, w))
            depth_mask[h_start:h_start + h_len , w_start:w_start + w_len] = 0
            flow_mask = np.ones((h, w, 2))
            flow_mask[h_start:h_start + h_len, w_start:w_start + w_len] = 0

            forward_img_mask = ForwardWarping()(img_mask, flow01, img0_depth, size)
            forward_depth_mask = ForwardWarping()(depth_mask, flow01, img0_depth, size)
            forward_flow_mask = ForwardWarping()(flow_mask, flow01, img0_depth, size)
            backward_img_mask = ForwardWarping()(img_mask, back_flow01, img1_depth, size)
            backward_depth_mask = ForwardWarping()(depth_mask, back_flow01, img1_depth, size)
            backward_flow_mask = ForwardWarping()(flow_mask, back_flow01, img1_depth, size)

        augment_img0 = img0 * img_mask
        augment_img0_depth = img0_depth * depth_mask
        augment_img0_depth = utils.fix_depth(augment_img0_depth, augment_img0)
        augment_img1 = img1 * img_mask
        augment_img1_depth = img1_depth * depth_mask
        augment_img1_depth = utils.fix_depth(augment_img1_depth, augment_img1)

        augment_img0_flow = flow01 * flow_mask
        augment_img1_flow = flow01 * backward_flow_mask
        back_augment_img0_flow = back_flow01 * forward_flow_mask
        back_augment_img1_flow = back_flow01 * flow_mask

        adjust_img0 = img0 * backward_img_mask
        adjust_img0_depth = img0_depth * backward_depth_mask
        adjust_img0_depth = utils.fix_depth(adjust_img0_depth, adjust_img0)
        adjust_img1 = img1 * forward_img_mask
        adjust_img1_depth = img1_depth * forward_depth_mask
        adjust_img1_depth = utils.fix_depth(adjust_img1_depth, adjust_img1)

        return (augment_img0, augment_img0_depth,
                augment_img0_flow, back_augment_img0_flow, 
                adjust_img1, adjust_img1_depth), (
                adjust_img0, adjust_img0_depth,
                augment_img1_flow, back_augment_img1_flow,
                augment_img1, augment_img1_depth)
    elif augment_flow_type >= 0.:
        normal_flow_type = utils.get_random(3, 0, False)
        if normal_flow_type >= 2.:
            gray = np.array([[0.2989, 0.2989, 0.2989],
                             [0.5870, 0.5870, 0.5870],
                             [0.1140, 0.1140, 0.1140]])
            augment_img_func = lambda img, g=gray: img @ g
        elif normal_flow_type >= 1.:
            channel = int(utils.get_random(3, 0, False))
            shift = np.array([utils.get_random(10, 15) if i == channel else 0 for i in range(3)])
            augment_img_func = lambda img, c=channel, s=shift: img + s
        elif normal_flow_type >= 0.:
            scale = utils.get_random(1, 0, False)
            augment_img_func = lambda img, s=scale: img * s

        augment_img0 = augment_img_func(img0)
        augment_img1 = augment_img_func(img1)
            
        return (augment_img0, img0_depth,
                flow01, back_flow01, 
                img1, img1_depth), (
                img0, img0_depth,
                flow01, back_flow01, 
                augment_img1, img1_depth)

    

if __name__ == "__main__":
    output_dir = "output/augment"
    input_dir = f"output/stereo/saves" 
    img0 = np.load(f"{input_dir}/img0.npy")
    img1 = np.load(f"{input_dir}/img1.npy")
    img2 = np.load(f"{input_dir}/img2.npy")
    img0_depth = np.load(f"{input_dir}/img0_depth.npy")
    img1_depth = np.load(f"{input_dir}/img1_depth.npy")
    img2_depth = np.load(f"{input_dir}/img2_depth.npy")
    flow01 = torch.load(f"{input_dir}/flow01.pt")
    flow12 = torch.load(f"{input_dir}/flow12.pt")
    flow02 = torch.load(f"{input_dir}/flow02.pt")
    back_flow01 = torch.load(f"{input_dir}/back_flow01.pt")
    back_flow12 = torch.load(f"{input_dir}/back_flow12.pt")
    back_flow02 = torch.load(f"{input_dir}/back_flow02.pt")
    h, w, c = img0.shape
    size = (h, w)
    

    set1, set2 = augment_flow(img0, img0_depth, img1, img1_depth,
                              flow01, back_flow01, size)
    (augment_img0, augment_img0_depth,
    augment_img0_flow, back_augment_img0_flow,
    img1, img1_depth) = set1
    (img0, img0_depth,
    augment_img1_flow, back_augment_img1_flow,
    augment_img1, augment_img1_depth) = set2


    output_dir_set1 = f"{output_dir}/set1"
    cv2.imwrite(f"{output_dir_set1}/augment_img0.png", augment_img0)
    plt.imsave(f"{output_dir_set1}/augment_img0_depth.png", 1 / augment_img0_depth, cmap="magma")
    cv2.imwrite(f"{output_dir_set1}/img1.png", img1)
    plt.imsave(f"{output_dir_set1}/img1_depth.png", 1 / img1_depth, cmap="magma")
    _, augment_img0_flow_color = utils.color_flow(augment_img0_flow)
    _, back_augment_img0_flow_color = utils.color_flow(back_augment_img0_flow)
    cv2.imwrite(f"{output_dir_set1}/augment_img0_flow.png", augment_img0_flow_color)
    cv2.imwrite(f"{output_dir_set1}/back_augment_img0_flow.png", back_augment_img0_flow_color)

    test = ForwardWarping()(augment_img0, augment_img0_flow, augment_img0_depth, size)
    cv2.imwrite(f"{output_dir_set1}/test.png", test)

    output_dir_set2 = f"{output_dir}/set2"
    cv2.imwrite(f"{output_dir_set2}/img0.png", img0)
    plt.imsave(f"{output_dir_set2}/img0_depth.png", 1 / img0_depth, cmap="magma")
    cv2.imwrite(f"{output_dir_set2}/augment_img1.png", augment_img1)
    plt.imsave(f"{output_dir_set2}/augment_img1_depth.png", 1 / augment_img1_depth, cmap="magma")
    _, augment_img1_flow_color = utils.color_flow(augment_img1_flow)
    _, back_augment_img1_flow_color = utils.color_flow(back_augment_img1_flow)
    cv2.imwrite(f"{output_dir_set2}/augment_img1_flow.png", augment_img1_flow_color)
    cv2.imwrite(f"{output_dir_set2}/back_augment_img1_flow.png", back_augment_img1_flow_color)


    test2 = ForwardWarping()(img0, augment_img1_flow, img0_depth, size)
    cv2.imwrite(f"{output_dir_set2}/test2.png", test2)












