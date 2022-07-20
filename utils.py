import geometry
import flow_colors

import cv2
import numpy as np
import torch
import random
import math

def get_img(path):
    img = cv2.imread(path, -1)
    if len(img) < 3:
        h, w = img.shape
        img = np.stack((img, img, img), -1)
    else:
        h, w, _ = img.shape
    return img, h, w

def get_stereo_img(path_left, path_right):
    left = cv2.imread(path_left, -1)
    right = cv2.imread(path_right, -1)
    if len(left) < 3:
        h, w = left.shape
        left = np.stack((left, left, left), -1)
        right = np.stack((right, right, right), -1)
    else:
        h, w, _ = left.shape
    return left, right, h, w

def get_depth(path, target_size, bits=8):
    depth_raw = cv2.imread(path, -1)
    # / (2 ** bits - 1)
    
    h, w = target_size
    if depth_raw.shape[0] != h or depth_raw.shape[1] != w:
        depth_raw = cv2.resize(depth_raw, (w, h))

    depth = depth_raw / (2 ** bits - 1)
    # depth = 1.0 / (depth + 0.005)
    # depth[depth > 100] = 100
    depth = normalize_depth(depth)

    return depth, depth_raw

def get_disparity(path, target_size, bits=8):
    disparity = cv2.imread(path, -1) / (2 ** bits - 1)
    
    h, w = target_size
    if disparity.shape[0] != h or disparity.shape[1] != w:
        disparity = cv2.resize(disparity, (w, h))
    
    disparity = disparity / w

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
    sign = ((-1) ** random.randrange(2)) if random_sign else 1
    value = (random.random() * random_range + random_begin)
    return sign * value

def normalize_depth(depth):
    depth = 1.0 / (depth + 0.005)
    depth[depth > 100] = 100
    return depth
    print(f"{np.max(disparity) = }")


def color_flow(flow):
    flow_16bit = np.concatenate((flow * 64. + (2 ** 15), np.ones_like(flow)[:, :, :, 0:1]), -1)[0]
    flow_16bit = cv2.cvtColor(flow_16bit, cv2.COLOR_BGR2RGB)
    flow_color = flow_colors.flow_to_color(flow[0].numpy(), convert_to_bgr=True)
    return flow_16bit, flow_color

def inpaint_img(img, masks):
    img = cv2.inpaint(img, 1 - masks["H"], 3, cv2.INPAINT_TELEA)
    return img


