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

from preprocess import Convert, flow_forward_warping
from bilateral_filter import sparse_bilateral_filtering
from using_clib import Resample2d

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

def augment_flow(img0, img0_depth, img0_mask, img1, img1_depth, img1_mask,
                 flow01, back_flow01, size):
    use_special_flow = utils.get_random(0, 1) > 0
    use_special_flow = 1
    if use_special_flow:
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

        augment_img0, _ = flow_forward_warping(img0, special_flow, img0_depth, size)
        augment_img0_depth, _ = flow_forward_warping(img0_depth, special_flow, img0_depth, size)
        augment_img0_depth = utils.fix_depth(augment_img0_depth, augment_img0)
        augment_img0_mask, _ = flow_forward_warping(img0_mask, special_flow, img0_depth, size)
        augment_img1, _ = flow_forward_warping(img1, special_flow, img1_depth, size)
        augment_img1_depth, _ = flow_forward_warping(img1_depth, special_flow, img1_depth, size)
        augment_img1_depth = utils.fix_depth(augment_img1_depth, augment_img1)
        augment_img1_mask, _ = flow_forward_warping(img1_mask, special_flow, img1_depth, size)

        back_augment_img0_flow, _ = Convert.flow_to_backward_flow(augment_img0_flow, augment_img0_depth, size)
        back_augment_img1_flow, _ = Convert.flow_to_backward_flow(augment_img1_flow, img0_depth, size)

        return (augment_img0, augment_img0_depth, augment_img0_mask,
                augment_img0_flow, back_augment_img0_flow, 
                img1, img1_depth, img1_mask), (
                img0, img0_depth, img0_mask,
                augment_img1_flow, back_augment_img1_flow,
                augment_img1, augment_img1_depth, augment_img1_mask)
    else:
        no_flow_type = utils.get_random(3, 0, False)
        # no_flow_type = 2.2
        if no_flow_type >= 2.:
            gray = np.array([[0.2989, 0.2989, 0.2989],
                             [0.5870, 0.5870, 0.5870],
                             [0.1140, 0.1140, 0.1140]])
            augment_func = lambda img, g=gray: img @ g
        elif no_flow_type >= 1.:
            channel = int(utils.get_random(3, 0, False))
            shift = np.array([utils.get_random(10, 15) if i == channel else 0 for i in range(3)])
            augment_func = lambda img, c=channel, s=shift: img + s
        elif no_flow_type >= 0.:
            scale = utils.get_random(1, 0, False)
            augment_func = lambda img, s=scale: img * s

        augment_img0 = augment_func(img0)
        print(f"{augment_img0[0, 0] = }")
        print(f"{img0[0, 0] = }")
        augment_img1 = augment_func(img1)
            
        return (augment_img0, img0_depth, img0_mask,
                flow01, back_flow01, 
                img1, img1_depth, img1_mask), (
                img0, img0_depth, img0_mask,
                flow01, back_flow01, 
                augment_img1, img1_depth, img1_mask)

    

if __name__ == "__main__":
    output_dir = "output/augment"
    input_dir = f"output/stereo/saves" 
    img0 = np.load(f"{input_dir}/img0.npy")
    img1 = np.load(f"{input_dir}/img1.npy")
    img2 = np.load(f"{input_dir}/img2.npy")
    img0_depth = np.load(f"{input_dir}/img0_depth.npy")
    img1_depth = np.load(f"{input_dir}/img1_depth.npy")
    img2_depth = np.load(f"{input_dir}/img2_depth.npy")
    img0_mask = np.load(f"{input_dir}/img0_mask.npy")
    img1_mask = np.load(f"{input_dir}/img1_mask.npy")
    img2_mask = np.load(f"{input_dir}/img2_mask.npy")
    flow01 = torch.load(f"{input_dir}/flow01.pt")
    flow12 = torch.load(f"{input_dir}/flow12.pt")
    flow02 = torch.load(f"{input_dir}/flow02.pt")
    back_flow01 = torch.load(f"{input_dir}/back_flow01.pt")
    back_flow12 = torch.load(f"{input_dir}/back_flow12.pt")
    back_flow02 = torch.load(f"{input_dir}/back_flow02.pt")
    h, w, c = img0.shape
    size = (h, w)
    

    set1, set2 = augment_flow(img0, img0_depth, img0_mask, img1, img1_depth, img1_mask,
                              flow01, back_flow01, size)
    (augment_img0, augment_img0_depth, augment_img0_mask,
    augment_img0_flow, back_augment_img0_flow,
    img1, img1_depth, img1_mask) = set1
    (img0, img0_depth, img0_mask,
    augment_img1_flow, back_augment_img1_flow,
    augment_img1, augment_img1_depth, augment_img1_mask) = set2


    output_dir_set1 = f"{output_dir}/set1"
    cv2.imwrite(f"{output_dir_set1}/augment_img0.png", augment_img0)
    plt.imsave(f"{output_dir_set1}/augment_img0_depth.png", 1 / augment_img0_depth, cmap="magma")
    cv2.imwrite(f"{output_dir_set1}/augment_img0_mask.png", augment_img0_mask * 255)
    cv2.imwrite(f"{output_dir_set1}/img1.png", img1)
    plt.imsave(f"{output_dir_set1}/img1_depth.png", 1 / img1_depth, cmap="magma")
    cv2.imwrite(f"{output_dir_set1}/img1_mask.png", img1_mask * 255)
    _, augment_img0_flow_color = utils.color_flow(augment_img0_flow)
    _, back_augment_img0_flow_color = utils.color_flow(back_augment_img0_flow)
    cv2.imwrite(f"{output_dir_set1}/augment_img0_flow.png", augment_img0_flow_color)
    cv2.imwrite(f"{output_dir_set1}/back_augment_img0_flow.png", back_augment_img0_flow_color)


    output_dir_set2 = f"{output_dir}/set2"
    cv2.imwrite(f"{output_dir_set2}/img0.png", img0)
    plt.imsave(f"{output_dir_set2}/img0_depth.png", 1 / img0_depth, cmap="magma")
    cv2.imwrite(f"{output_dir_set2}/img0_mask.png", img0_mask * 255)
    cv2.imwrite(f"{output_dir_set2}/augment_img1.png", augment_img1)
    plt.imsave(f"{output_dir_set2}/augment_img1_depth.png", 1 / augment_img1_depth, cmap="magma")
    cv2.imwrite(f"{output_dir_set2}/augment_img1_mask.png", augment_img1_mask * 255)
    _, augment_img1_flow_color = utils.color_flow(augment_img1_flow)
    _, back_augment_img1_flow_color = utils.color_flow(back_augment_img1_flow)
    cv2.imwrite(f"{output_dir_set2}/augment_img1_flow.png", augment_img1_flow_color)
    cv2.imwrite(f"{output_dir_set2}/back_augment_img1_flow.png", back_augment_img1_flow_color)














