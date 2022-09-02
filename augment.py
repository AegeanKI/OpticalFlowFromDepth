import utils
import preprocess
import numpy as np
import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import cv2
import matplotlib.pyplot as plt
import pykitti
# from parser import DataParser

from preprocess import Convert, ConcatFlow, BackFlow
from bilateral_filter import sparse_bilateral_filtering
from using_clib import Resample2d2
# from using_clib import ForwardWarping
import glob
import os

class SpecialFlow(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.h, self.w = size

        meshgrid = torch.meshgrid(torch.arange(self.w), torch.arange(self.h), indexing="xy")
        self.p0 = torch.stack(meshgrid, axis=-1).type(torch.float64).to("cuda")
        # meshgrid = np.meshgrid(range(self.w), range(self.h), indexing="xy")
        # self.p0 = np.stack(meshgrid, axis=-1).astype(np.float64)

    def forward(self, augment_flow_type):
        if augment_flow_type >= 7.:
            p1, p_prev = self._shear()
        elif augment_flow_type >= 6.:
            p1, p_prev = self._rotate()
        elif augment_flow_type >= 5.:
            p1, p_prev = self._flip()

        print(f"{p1.get_device() = }")
        print(f"{p_prev.get_device() = }")

        special_flow = (p1 - self.p0).permute(2, 0, 1).unsqueeze(0)
        back_special_flow = (self.p0 - p_prev).permute(2, 0, 1).unsqueeze(0)
        return special_flow, back_special_flow

    def _flip(self):
        horizontal = utils.get_random(0, 1) > 0

        if horizontal:
            meshgrid = torch.meshgrid(torch.arange(self.w - 1, -1, -1), torch.arange(self.h), indexing="xy")
        else:
            meshgrid = torch.meshgrid(torch.arange(self.w), torch.arange(self.h - 1, -1, -1), indexing="xy")

        p1 = torch.stack(meshgrid, axis=-1).type(torch.float64).to("cuda")
        p_prev = p1
        return p1, p_prev

    def _rotate(self):
        c0 = (utils.get_random(3 * self.w, -self.w), utils.get_random(3 * self.h, -self.h))
        c0 = torch.tensor(c0).to("cuda")
        theta = torch.deg2rad(torch.tensor(utils.get_random(10, 15)))

        rotate = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]]).type(torch.float64).to("cuda")
        reverse_rotate = torch.tensor([[torch.cos(-theta), -torch.sin(-theta)],
                                       [torch.sin(-theta), torch.cos(-theta)]]).type(torch.float64).to("cuda")

        p1 = (self.p0 - c0) @ rotate + c0
        p_prev = (self.p0 - c0) @ reverse_rotate + c0
        return p1, p_prev

    def _shear(self):
        horizontal = utils.get_random(0, 1) > 0
        shear_range = utils.get_random(0.10, 0.15)

        if horizontal:
            shear = torch.tensor([[1, 0], [shear_range, 1]]).type(torch.float64).to("cuda")
            reverse_shear = torch.tensor([[1, 0], [-shear_range, 1]]).type(torch.float64).to("cuda")
        else:
            shear = torch.tensor([[1, shear_range], [0, 1]]).type(torch.float64).to("cuda")
            reverse_shear = torch.tensor([[1, -shear_range], [0, 1]]).type(torch.float64).to("cuda")
        
        p1 = self.p0 @ shear
        p_prev = self.p0 @ reverse_shear
        return p1, p_prev

class AugmentFlow(nn.Module):
    def __init__(self, size, batch_size, device="cuda"):
        super().__init__()
        self.size = size
        self.h, self.w = size
        self.batch_size = batch_size
        self.sf = SpecialFlow(size).to(device)

    def forward(self, img0, img0_depth, img1, img1_depth, flow01, back_flow01):
        augment_flow_type = utils.get_random(8, 0, False)
        augment_flow_type = 5.2
        if augment_flow_type >= 5.:
            special_flow, back_special_flow = self.sf(augment_flow_type)
            special_flow = special_flow.repeat(self.batch_size, 1, 1, 1)



def augment_flow(img0, img0_depth, img1, img1_depth,
                 flow01, back_flow01, size, batch_size, device):
    h, w = size
    augment_flow_type = utils.get_random(8, 0, False)
    augment_flow_type = 7.2
    re = Resample2d2(size, batch_size)
    if augment_flow_type >= 5.:
        sf = SpecialFlow(size).to(device)
        special_flow, back_special_flow = sf(augment_flow_type)
        special_flow = special_flow.to(device)
        back_special_flow = back_special_flow.to(device)
        
        print(f"{special_flow.get_device() = }")

        special_flow = special_flow.repeat(batch_size, 1, 1, 1)
        back_special_flow = back_special_flow.repeat(batch_size, 1, 1, 1)

        # augment_img0_flow = Convert.two_contiguous_flows_to_one_flow(special_flow, back_special_flow,
        #                                                              flow01, img0_depth, size)
        cf = ConcatFlow(size, batch_size).to("cuda")
        augment_img0_flow = cf(special_flow, back_special_flow, flow01, img0_depth)
        augment_img1_flow = cf(flow01, back_flow01, special_flow, img1_depth)

        # augment_img1_flow = Convert.two_contiguous_flows_to_one_flow(flow01, back_flow01,
        #                                                              special_flow, img1_depth, size)

        augment_img0 = re(img0, special_flow, img0_depth)
        augment_img0_depth = re(img0_depth, special_flow, img0_depth)
        augment_img0_depth = utils.fix_depth(augment_img0_depth, augment_img0)
        augment_img1 = re(img1, special_flow, img1_depth)
        augment_img1_depth = re(img1_depth, special_flow, img1_depth)
        augment_img1_depth = utils.fix_depth(augment_img1_depth, augment_img1)

        bf = BackFlow(size, batch_size).to("cuda")
        # back_augment_img0_flow = Convert.flow_to_backward_flow(augment_img0_flow, augment_img0_depth, size)
        # back_augment_img1_flow = Convert.flow_to_backward_flow(augment_img1_flow, img0_depth, size)
        back_augment_img0_flow = bf(augment_img0_flow, augment_img0_depth)
        back_augment_img1_flow = bf(augment_img1_flow, img0_depth)

        return (augment_img0, augment_img0_depth,
                augment_img0_flow, back_augment_img0_flow, 
                img1, img1_depth), (
                img0, img0_depth,
                augment_img1_flow, back_augment_img1_flow,
                augment_img1, augment_img1_depth), int(augment_flow_type)
    elif augment_flow_type >= 3.:
        if augment_flow_type >= 4.:
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

            forward_img_mask = re(img_mask, flow01, img0_depth)
            forward_depth_mask = re(depth_mask, flow01, img0_depth)
            forward_flow_mask = re(flow_mask, flow01, img0_depth)
            backward_img_mask = re(img_mask, back_flow01, img1_depth)
            backward_depth_mask = re(depth_mask, back_flow01, img1_depth)
            backward_flow_mask = re(flow_mask, back_flow01, img1_depth)
        elif augment_flow_type >= 3.:
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

            forward_img_mask = re(img_mask, flow01, img0_depth)
            forward_depth_mask = re(depth_mask, flow01, img0_depth)
            forward_flow_mask = re(flow_mask, flow01, img0_depth)
            backward_img_mask = re(img_mask, back_flow01, img1_depth)
            backward_depth_mask = re(depth_mask, back_flow01, img1_depth)
            backward_flow_mask = re(flow_mask, back_flow01, img1_depth)

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

        adjust = utils.get_random(0, 1) > 0
        adjust_img0, adjust_img0_depth = img0, img0_depth
        adjust_img1, adjust_img1_depth = img1, img1_depth
        if adjust:
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
                augment_img1, augment_img1_depth), int(augment_flow_type)
    elif augment_flow_type >= 0.:
        if augment_flow_type >= 2.:
            gray = np.array([[0.2989, 0.2989, 0.2989],
                             [0.5870, 0.5870, 0.5870],
                             [0.1140, 0.1140, 0.1140]])
            augment_img_func = lambda img, g=gray: img @ g
        elif augment_flow_type >= 1.:
            channel = int(utils.get_random(3, 0, False))
            shift = np.array([utils.get_random(10, 15) if i == channel else 0 for i in range(3)])
            augment_img_func = lambda img, c=channel, s=shift: img + s
        elif augment_flow_type >= 0.:
            scale = utils.get_random(1, 0, False)
            augment_img_func = lambda img, s=scale: img * s

        augment_img0 = augment_img_func(img0)
        augment_img1 = augment_img_func(img1)
            
        return (augment_img0, img0_depth,
                flow01, back_flow01, 
                img1, img1_depth), (
                img0, img0_depth,
                flow01, back_flow01, 
                augment_img1, img1_depth), int(augment_flow_type)


def augment(img0, img0_depth, img1, img1_depth, flow01, back_flow01, size, output_dir, batch_size, device):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    set1, set2, augment_flow_type = augment_flow(img0, img0_depth, img1, img1_depth,
                                                 flow01, back_flow01, size, batch_size, device)
    return

    (augment_img0, augment_img0_depth,
    augment_img0_flow, back_augment_img0_flow,
    img1, img1_depth) = set1
    (img0, img0_depth,
    augment_img1_flow, back_augment_img1_flow,
    augment_img1, augment_img1_depth) = set2

    # output_dir_set1 = f"{output_dir}/set1"
    # if not os.path.exists(output_dir_set1):
    #     os.mkdir(output_dir_set1)
    
    # np.save(f"{output_dir_set1}/augment_img0.npy", augment_img0)
    # np.save(f"{output_dir_set1}/augment_img0_depth.npy", augment_img0_depth)
    # np.save(f"{output_dir_set1}/img1.npy", img1)
    # np.save(f"{output_dir_set1}/img1_depth.npy", img1_depth)
    # cv2.imwrite(f"{output_dir_set1}/augment_img0.png", augment_img0)
    # plt.imsave(f"{output_dir_set1}/augment_img0_depth.png", 1 / augment_img0_depth, cmap="magma")
    # cv2.imwrite(f"{output_dir_set1}/img1.png", img1)
    # plt.imsave(f"{output_dir_set1}/img1_depth.png", 1 / img1_depth, cmap="magma")
    # _, augment_img0_flow_color = utils.color_flow(augment_img0_flow)
    # _, back_augment_img0_flow_color = utils.color_flow(back_augment_img0_flow)
    # torch.save(augment_img0_flow, f"{output_dir_set1}/augment_img0_flow.pt")
    # torch.save(back_augment_img0_flow, f"{output_dir_set1}/back_augment_img0_flow.pt")
    # cv2.imwrite(f"{output_dir_set1}/augment_img0_flow.png", augment_img0_flow_color)
    # cv2.imwrite(f"{output_dir_set1}/back_augment_img0_flow.png", back_augment_img0_flow_color)

    # output_dir_set2 = f"{output_dir}/set2"
    # if not os.path.exists(output_dir_set2):
    #     os.mkdir(output_dir_set2)
    # np.save(f"{output_dir_set2}/img0.npy", img0)
    # np.save(f"{output_dir_set2}/img0_depth.npy", img0_depth)
    # np.save(f"{output_dir_set2}/augment_img1.npy", augment_img1)
    # np.save(f"{output_dir_set2}/augment_img1_depth.npy", augment_img1_depth)
    # cv2.imwrite(f"{output_dir_set2}/img0.png", img0)
    # plt.imsave(f"{output_dir_set2}/img0_depth.png", 1 / img0_depth, cmap="magma")
    # cv2.imwrite(f"{output_dir_set2}/augment_img1.png", augment_img1)
    # plt.imsave(f"{output_dir_set2}/augment_img1_depth.png", 1 / augment_img1_depth, cmap="magma")
    # _, augment_img1_flow_color = utils.color_flow(augment_img1_flow)
    # _, back_augment_img1_flow_color = utils.color_flow(back_augment_img1_flow)
    # torch.save(augment_img0_flow, f"{output_dir_set2}/augment_img1_flow.pt")
    # torch.save(back_augment_img0_flow, f"{output_dir_set2}/back_augment_img1_flow.pt")
    # cv2.imwrite(f"{output_dir_set2}/augment_img1_flow.png", augment_img1_flow_color)
    # cv2.imwrite(f"{output_dir_set2}/back_augment_img1_flow.png", back_augment_img1_flow_color)

    

if __name__ == "__main__":
    from cuda.fw import FW

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float64
    img_dirs = glob.glob("output/monocular/*")
      
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device = }")
    for idx, img_dir in enumerate(img_dirs):
        img_dir = "output/monocular/000000118762.jpg"

        img0 = np.load(f"{img_dir}/img0.npy")
        img1 = np.load(f"{img_dir}/img1.npy")
        img2 = np.load(f"{img_dir}/img2.npy")
        img0_depth = np.load(f"{img_dir}/img0_depth.npy")
        img1_depth = np.load(f"{img_dir}/img1_depth.npy")
        img2_depth = np.load(f"{img_dir}/img2_depth.npy")
        flow01 = torch.load(f"{img_dir}/flow01.pt")
        flow12 = torch.load(f"{img_dir}/flow12.pt")
        flow02 = torch.load(f"{img_dir}/flow02.pt")
        back_flow01 = torch.load(f"{img_dir}/back_flow01.pt")
        back_flow12 = torch.load(f"{img_dir}/back_flow12.pt")
        back_flow02 = torch.load(f"{img_dir}/back_flow02.pt")

        cv2.imwrite(f"testing/img0.png", img0)
        cv2.imwrite(f"testing/img1.png", img1)
        cv2.imwrite(f"testing/img2.png", img2)
        _, flow02_color = utils.color_flow(flow02)
        cv2.imwrite(f"testing/flow02.png", flow02_color)
    
        h, w, c = img0.shape
        size = (h, w)

        img0 = torch.from_numpy(img0).permute(2, 0, 1).type(torch.float64)
        img1 = torch.from_numpy(img1).permute(2, 0, 1).type(torch.float64)
        img2 = torch.from_numpy(img2).permute(2, 0, 1).type(torch.float64)
        img0_depth = torch.from_numpy(img0_depth).unsqueeze(0).type(torch.float64)
        img1_depth = torch.from_numpy(img1_depth).unsqueeze(0).type(torch.float64)
        img2_depth = torch.from_numpy(img2_depth).unsqueeze(0).type(torch.float64)
        flow01 = flow01.squeeze().permute(2, 0, 1).type(torch.float64)
        flow12 = flow12.squeeze().permute(2, 0, 1).type(torch.float64)
        flow02 = flow02.squeeze().permute(2, 0, 1).type(torch.float64)
        back_flow01 = back_flow01.squeeze().permute(2, 0, 1).type(torch.float64)
        back_flow12 = back_flow12.squeeze().permute(2, 0, 1).type(torch.float64)
        back_flow02 = back_flow02.squeeze().permute(2, 0, 1).type(torch.float64)

        batch_size = 1
        img0 = img0.repeat(batch_size, 1, 1, 1).to(device)
        img1 = img1.repeat(batch_size, 1, 1, 1).to(device)
        img2 = img2.repeat(batch_size, 1, 1, 1).to(device)
        img0_depth = img0_depth.repeat(batch_size, 1, 1, 1).to(device)
        img1_depth = img1_depth.repeat(batch_size, 1, 1, 1).to(device)
        img2_depth = img2_depth.repeat(batch_size, 1, 1, 1).to(device)
        flow01 = flow01.repeat(batch_size, 1, 1, 1).to(device)
        flow12 = flow12.repeat(batch_size, 1, 1, 1).to(device)
        flow02 = flow02.repeat(batch_size, 1, 1, 1).to(device)
        back_flow01 = back_flow01.repeat(batch_size, 1, 1, 1).to(device)
        back_flow12 = back_flow12.repeat(batch_size, 1, 1, 1).to(device)
        back_flow02 = back_flow02.repeat(batch_size, 1, 1, 1).to(device)

        print(f"{img0.shape = }, {img0.dtype = }")
        print(f"{img0_depth.shape = }, {img0_depth.dtype = }")
        print(f"{flow01.shape = }, {flow01.dtype = }")
        print(f"{back_flow01.shape = }, {back_flow01.dtype = }")
        print(f"{batch_size = }")
        print(f"{h = }")
        print(f"{w = }")
        print("==============================\n")

        cf = ConcatFlow(size, batch_size).to("cuda")
        test_flow = cf(flow01, back_flow01, flow12, img1_depth) 
        test_flow_color = test_flow.cpu().permute(0, 2, 3, 1)
        _, test_flow_color = utils.color_flow(test_flow_color)
        cv2.imwrite(f"testing/test_flow.png", test_flow_color)

        # concat_flow = Convert.two_contiguous_flows_to_one_flow(flow01, back_flow01, flow12, img1_depth, size)

        fw = Resample2d2(size, batch_size).to("cuda")
        test_img = fw(img0, test_flow, img0_depth)
        test_img = test_img[0].permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(f"testing/test_img.png", test_img)
        # test2_img = fw(img0, flow02, img0_depth)
        # test2_img = test2_img[0].permute(1, 2, 0).cpu().numpy()
        # cv2.imwrite(f"testing/test2_img.png", test2_img)



        break

        img_name = img_dir.split("/")[-1]

        # if not os.path.exists(f"testing/{img_name}"):
        #     os.mkdir(f"testing/{img_name}")
        print(f"{img_name} {idx + 1} / {len(img_dirs)}")
        augment(img0, img0_depth, img1, img1_depth, flow01, back_flow01, size, f"testing/{img_name}/01", batch_size, device)
        # augment(img1, img1_depth, img2, img2_depth, flow12, back_flow12, size, f"output/augment/{img_name}/12", batch_size)
        # augment(img0, img0_depth, img2, img2_depth, flow02, back_flow02, size, f"output/augment/{img_name}/02", batch_size)


        break








