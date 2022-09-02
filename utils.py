import geometry
import flow_colors

import cv2
import numpy as np
import torch
import random
import math
import skimage
from bilateral_filter import sparse_bilateral_filtering

def get_img(path):
    img = cv2.imread(path, -1)
    if img.ndim == 2:
        h, w = img.shape
        img = torch.from_numpy(img).type(torch.float64).repeat(3, 1, 1)
    else:
        h, w, _ = img.shape
        img = torch.from_numpy(img).type(torch.float64).permute(2, 0, 1)
    return img, (h, w) # CHW, (int, int), C = 3

def get_stereo_img(path_left, path_right):
    left = cv2.imread(path_left, -1)
    right = cv2.imread(path_right, -1)
    if left.ndim == 2:
        h, w = left.shape
        left = torch.from_numpy(left).type(torch.float64).repeat(3, 1, 1)
        right = torch.from_numpy(right).type(torch.float64).repeat(3, 1, 1)
    else:
        h, w, _ = left.shape
        left = torch.from_numpy(left).type(torch.float64).permute(2, 0, 1)
        right = torch.from_numpy(right).type(torch.float64).permute(2, 0, 1)

    return left, right, (h, w) # CHW, CHW, (int, int), C = 1

def get_depth(path, target_size, bits=8):
    # depth = cv2.imread(path, -1) / (2 ** bits - 1)
    depth = cv2.imread(path, -1) / (2 ** bits)
    
    h, w = target_size
    if depth.shape[0] != h or depth.shape[1] != w:
        depth = cv2.resize(depth, (w, h))

    depth = normalize_depth(depth)
    depth = torch.from_numpy(depth).unsqueeze(0)
    return depth # CHW, C = 1

def get_disparity(path, target_size, bits=8):
    # disparity = cv2.imread(path, -1) / (2 ** bits - 1)
    disparity = cv2.imread(path, -1) / (2 ** bits)
    
    disparity = skimage.measure.block_reduce(disparity, (5, 5), np.max)

    h, w = target_size
    if disparity.shape[0] != h or disparity.shape[1] != w:
        disparity = cv2.resize(disparity, (w, h))
    
    # disparity = disparity / w

    return disparity


def get_mask(path, target_size):
    instances_mask = cv2.imread(path, -1)

    h, w = target_size
    if instances_mask.shape[0] != h or instances_mask.shape[1] != w:
        instances_mask = cv2.resize(instance_mask, (w, h))
    
    classes = instances_mask.max()
    areas = np.array([instances_mask[instances_mask==c].sum()
                      for c in range(classes+1)], np.float32)

    instances = []
    num_objects = 1
    labels = areas.argsort()[-num_objects:][::-1] if areas.shape[0] > 1 else []
    
    for l in labels:
        seg_mask = np.zeros_like(instances_mask)
        seg_mask[instances_mask == l] = 1
        seg_mask = np.expand_dims(seg_mask, 0)

        seg_mask = torch.from_numpy(np.stack((seg_mask, seg_mask), -1)).float()
        instances.append(seg_mask)

    return instances_mask, instances, labels
    
def get_random(random_range, random_begin, random_sign=True):
    sign = torch.randint(0, 2, (1,))[0] * 2 - 1 if random_sign else torch.tensor(1)
    # value = (random.random() * random_range + random_begin)
    value = torch.rand(1)[0] * random_range + torch.tensor(random_begin)
    return sign * value

def normalize_depth(depth):
    depth = 1.0 / (depth + 0.005)
    depth[depth > 100] = 100
    return depth


def color_flow(flow):
    flow_color = flow_colors.flow_to_color(flow[0].numpy(), convert_to_bgr=True)
    return None, flow_color
    flow_16bit = np.concatenate((flow * 64. + (2 ** 15), np.ones_like(flow)[:, :, :, 0:1]), -1)[0]
    flow_16bit = cv2.cvtColor(flow_16bit, cv2.COLOR_BGR2RGB)
    flow_color = flow_colors.flow_to_color(flow[0].numpy(), convert_to_bgr=True)
    return flow_16bit, flow_color


def inpaint_img(img, masks):
    img = cv2.inpaint(img, 1 - masks["H"], 3, cv2.INPAINT_TELEA)
    return img


def fix_depth(img_depth, img):
    # print(f"{img_depth.shape = }")
    # print(f"{img.shape = }")
    img_depth[img_depth == 0] = 100
    # img_depth = img_depth.detach().cpu().numpy()
    # img = img.detach().cpu().numpy()

    # for i in range(img_depth.shape[0]):
    #     img_depth[i, 0] = sparse_bilateral_filtering(img_depth[i, 0].copy(), img[i].copy(),
    #                                                  filter_size=[5, 5], num_iter=2)
    return img_depth


