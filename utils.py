import geometry
import flow_colors

import cv2
import numpy as np
import torch
import random
import math
import skimage
from bilateral_filter import sparse_bilateral_filtering
from threading import Thread
import os
# import pickle
import time


def get_img(path):
    img = cv2.imread(path, -1)
    # if img.ndim == 2:
    #     h, w = img.shape
    #     img = torch.from_numpy(img).type(torch.float32).repeat(3, 1, 1)
    # else:
    h, w, _ = img.shape
    img = torch.from_numpy(img).type(torch.float32).permute(2, 0, 1)
    return img, (h, w) # CHW, (int, int), C = 3

def get_stereo_img(path_left, path_right):
    left = cv2.imread(path_left, -1)
    right = cv2.imread(path_right, -1)
    
    h, w = 375, 1242
    if left.shape[0] != h or left.shape[1] != w:
        left = cv2.resize(left, (w, h))
        right = cv2.resize(right, (w, h))

    if left.ndim == 2:
        h, w = left.shape
        left = torch.from_numpy(left).type(torch.float32).repeat(3, 1, 1)
        right = torch.from_numpy(right).type(torch.float32).repeat(3, 1, 1)
    else:
        h, w, _ = left.shape
        left = torch.from_numpy(left).type(torch.float32).permute(2, 0, 1)
        right = torch.from_numpy(right).type(torch.float32).permute(2, 0, 1)

    return left, right, (h, w) # CHW, CHW, (int, int), C = 1

def get_depth(path, normalize=True):
    depth = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    depth = 1.0 / (depth + 0.005)
    if normalize:
        depth = normalize_depth(depth)

    h, w = depth.shape
    depth = torch.from_numpy(depth).unsqueeze(0)

    return depth, (h, w) # CHW, C = 1

def get_disparity(path):
    disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # print(f"{disp.min() = }")
    # print(f"{disp.max() = }")

    disp[disp == np.inf] = 0
    # disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    disp = disp * 63 / 255
    # disp = 255 / disp
    # .astype(np.uint8)
    h, w = disp.shape
    disp = torch.from_numpy(disp).unsqueeze(0)
    
    return disp, (h, w) 

# def get_disparity(path, target_size, bits=8):
#     # disparity = cv2.imread(path, -1) / (2 ** bits - 1)
#     disparity = cv2.imread(path, -1) / (2 ** bits)
    
#     disparity = skimage.measure.block_reduce(disparity, (5, 5), np.max)

#     h, w = target_size
#     if disparity.shape[0] != h or disparity.shape[1] != w:
#         disparity = cv2.resize(disparity, (w, h))
   
#     disparity = torch.from_numpy(disparity).unsqueeze(0)
    
#     # disparity = disparity / w
#     del h, w
#     return disparity

def one_hot(idx, dim):
    res = torch.zeros(dim)
    res[idx] = 1
    return res


def get_random(random_range, random_begin, random_sign=True):
    sign = torch.randint(0, 2, (1,))[0] * 2 - 1 if random_sign else torch.tensor(1)
    # value = (random.random() * random_range + random_begin)
    value = torch.rand(1)[0] * random_range + torch.tensor(random_begin)
    return sign * value

def normalize_depth(depth):
    depth[depth == 0] = 100
    depth[depth > 100] = 100

    depth_min = depth.min()
    depth[depth == 100] = 0
    depth_max = depth.max()
    depth = (depth - depth_min) * 98 / (depth_max - depth_min) + 1
    depth[depth == ((0 - depth_min) * 98 / (depth_max - depth_min) + 1)] = 100

    # depth_min = depth.min()
    # depth_max = depth.max()
    
    # depth = (depth - depth_min) * 0.999 / (depth_max - depth_min) + 0.001    # [0.001, 1]
    return depth

def fix_warped_depth(depth):
    depth[depth == 0] = 100
    depth[depth > 99.5] = 100
    return depth

def color_flow(flow):
    flow_color = flow_colors.flow_to_color(flow[0].numpy(), convert_to_bgr=True)
    return None, flow_color
    flow_16bit = np.concatenate((flow * 64. + (2 ** 15), np.ones_like(flow)[:, :, :, 0:1]), -1)[0]
    flow_16bit = cv2.cvtColor(flow_16bit, cv2.COLOR_BGR2RGB)
    flow_color = flow_colors.flow_to_color(flow[0].numpy(), convert_to_bgr=True)
    return flow_16bit, flow_color


# def inpaint_img(img, masks):
#     img = cv2.inpaint(img, 1 - masks["H"], 3, cv2.INPAINT_TELEA)
#     return img

def inpaint(img, valid, collision):
    H = valid[0].cpu().numpy()
    M = collision[0].cpu().numpy()
    M = (1 - (H == M)).astype(np.uint8)
    M_prime = cv2.dilate(M, kernel=np.ones((3, 3,), np.uint8), iterations=1)
    P = (M_prime == M).astype(np.uint8)
    H_prime = H * P
    # cv2.imwrite(f"test_output/test_masks/H.png", H * 255)
    # cv2.imwrite(f"test_output/test_masks/M.png", M * 255)
    # cv2.imwrite(f"test_output/test_masks/M_prime.png", M_prime * 255)
    # cv2.imwrite(f"test_output/test_masks/P.png", P * 255)
    # cv2.imwrite(f"test_output/test_masks/H_prime.png", H_prime * 255)
    img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    inpaint_img = cv2.inpaint(img_np, 1 - H_prime, 3, cv2.INPAINT_TELEA)
    if inpaint_img.ndim == 2:
        inpaint_img = torch.tensor(inpaint_img).unsqueeze(0).to(img.get_device())
    else:
        inpaint_img = torch.tensor(inpaint_img).permute(2, 0, 1).to(img.get_device())
    return inpaint_img

# def batch_save_data(batch_idx, output_dir, to_save_data):
#     get_np_start_time = time.time()
    
#     to_save_data_np = to_save_data.detach().cpu().numpy()

#     get_np_end_time = time.time()
#     print(f"    get np time = {get_np_end_time - get_np_start_time}")
#     save_np_start_time = time.time()

#     np.savez_compressed(f"{output_dir}/{batch_idx}.npz",
#             img_depth_flow=to_save_data_np)

#     # np.savez_compressed(f"{output_dir}/{batch_idx}.npz",
#     #         img_depth_flow=to_save_data_np,
#     #         original_size=original_size_np)

#     save_np_end_time = time.time()
#     print(f"    save np time = {save_np_end_time - save_np_start_time}")

#     del to_save_data_np
#     # del original_size_np
#     del get_np_start_time, get_np_end_time
#     del save_np_start_time, save_np_end_time


def set_seed(seed=42, loader=None):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass
