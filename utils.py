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
    if img.ndim == 2:
        h, w = img.shape
        img = torch.from_numpy(img).type(torch.float32).repeat(3, 1, 1)
    else:
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

def get_depth(path, target_size, bits=8):
    # depth = cv2.imread(path, -1) / (2 ** bits - 1)
    depth = cv2.imread(path, -1) / (2 ** bits)
    
    h, w = target_size
    if depth.shape[0] != h or depth.shape[1] != w:
        depth = cv2.resize(depth, (w, h))

    depth = normalize_depth(depth)
    depth = torch.from_numpy(depth).unsqueeze(0)

    del h, w
    return depth # CHW, C = 1

def get_disparity(path, target_size, bits=8):
    # disparity = cv2.imread(path, -1) / (2 ** bits - 1)
    disparity = cv2.imread(path, -1) / (2 ** bits)
    
    disparity = skimage.measure.block_reduce(disparity, (5, 5), np.max)

    h, w = target_size
    if disparity.shape[0] != h or disparity.shape[1] != w:
        disparity = cv2.resize(disparity, (w, h))
   
    disparity = torch.from_numpy(disparity).unsqueeze(0)
    
    # disparity = disparity / w
    del h, w
    return disparity


# def get_mask(path, target_size):
#     instances_mask = cv2.imread(path, -1)

#     h, w = target_size
#     if instances_mask.shape[0] != h or instances_mask.shape[1] != w:
#         instances_mask = cv2.resize(instance_mask, (w, h))
    
#     classes = instances_mask.max()
#     areas = np.array([instances_mask[instances_mask==c].sum()
#                       for c in range(classes+1)], np.float32)

#     instances = []
#     num_objects = 1
#     labels = areas.argsort()[-num_objects:][::-1] if areas.shape[0] > 1 else []
    
#     for l in labels:
#         seg_mask = np.zeros_like(instances_mask)
#         seg_mask[instances_mask == l] = 1
#         seg_mask = np.expand_dims(seg_mask, 0)

#         seg_mask = torch.from_numpy(np.stack((seg_mask, seg_mask), -1)).float()
#         instances.append(seg_mask)

#     return instances_mask, instances, labels
    
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


def fix_depth(depth, img):
    # print(f"{img_depth.shape = }")
    # print(f"{img.shape = }")
    depth[depth == 0] = 100
    # img_depth = img_depth.detach().cpu().numpy()
    # img = img.detach().cpu().numpy()

    # for i in range(img_depth.shape[0]):
    #     img_depth[i, 0] = sparse_bilateral_filtering(img_depth[i, 0].copy(), img[i].copy(),
    #                                                  filter_size=[5, 5], num_iter=2)
    return depth
    # return multithread_fix_depth(depth, img)

def multithread_fix_depth(depth, img):
    depth[depth == 0] = 100
    depth[depth > 100] = 100
    b, _, h, w = depth.shape
    # img_depth = torch.zeros((b, 1, h, w)) 
    img_depth = torch.zeros_like(depth)
    threads = []
    for idx in range(b):
        thread = MultiThreadFixDepth(idx, depth[idx, 0].cpu().numpy(),
                                     img[idx].permute(1, 2, 0).cpu().numpy(), img_depth)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    depth_min = torch.min(depth)
    depth[depth == 100] = 0
    depth_max = torch.max(depth)
    depth = (depth - depth_min) * 98 / (depth_max - depth_min) + 1
    depth[depth == (-depth_min * 98 / (depth_max - depth_min) + 1)] = 100

    
    del thread, threads
    del b, h, w
    return img_depth


class MultiThreadFixDepth(Thread):
    def __init__(self, idx, depth, img, img_depth):
        super().__init__()
        self.idx = idx
        self.depth = depth
        self.img = img
        self.img_depth = img_depth

    def run(self):
        fixed_depth = sparse_bilateral_filtering(self.depth.copy(), self.img.copy(),
                                                     filter_size=[5, 5], num_iter=2)
        self.img_depth[self.idx, 0] = torch.from_numpy(fixed_depth)



class MutliThreadLoad(Thread):
    def __init__(self, idx, imgs_and_depths, original_size, img_name):
        super().__init__()
        self.idx = idx
        self.imgs_and_depths = imgs_and_depths
        self.original_size = original_size
        self.img_name = img_name

    def run(self):
        img_depth_name = self.img_name.replace("jpg", "png")
        img_full_path = f"input/monocular/image/{self.img_name}"
        depth_full_path = f"input/monocular/depth/{img_depth_name}"
        img, size = get_img(img_full_path)
        img_depth  = get_depth(depth_full_path, size, 16) # depth.png: 0 ~ 65535
        img_depth = fix_depth(img_depth, img)
        self.imgs_and_depths[self.idx] = (img, img_depth)
        self.original_size[self.idx] = size

        del img, img_depth, size
        del img_full_path, depth_full_path


class MultiThreadFill(Thread):
    def __init__(self, idx, img0, img0_depth, img, depth):
        super().__init__()
        self.idx = idx
        self.img0 = img0
        self.img0_depth = img0_depth
        self.img = img
        self.depth = depth

    def run(self):
        _, h, w = self.img.shape
        self.img0[self.idx, :, :h, :w] = self.img
        self.img0_depth[self.idx, :, :h, :w] = self.depth

        del self.img, self.depth
        del h, w


class MultiThreadSave(Thread):
    # def __init__(self, idx, img_name, size, output_dir, img0, img1, img2, img0_depth, img1_depth, img2_depth,
    #              concat_img, flow01, flow12, flow02, back_flow01, back_flow12, back_flow02):
    def __init__(self, idx, img_name, size, output_dir, to_save_data):
        super().__init__()
        self.idx = idx
        self.img_name = img_name
        self.size = size
        self.output_dir = output_dir
        # self.img0 = img0
        # self.img1 = img1
        # self.img2 = img2
        # self.img0_depth = img0_depth
        # self.img1_depth = img1_depth
        # self.img2_depth = img2_depth
        # self.concat_img = concat_img
        # self.flow01 = flow01
        # self.flow12 = flow12
        # self.flow02 = flow02
        # self.back_flow01 = back_flow01
        # self.back_flow12 = back_flow12
        # self.back_flow02 = back_flow02
        self.to_save_data = to_save_data

    def run(self):
        # if not os.path.exists(output_dir):
        # os.mkdir(output_dir)
        h, w = self.size
        save_visual_data = False
        # save_visual_data = True

        if not save_visual_data:
            thread_start_np_time = time.time()
            
            to_save_data_np = self.to_save_data[self.idx, :, :h, :w].detach().cpu().numpy()
            
            thread_end_np_time = time.time()
            print(f"thread {self.idx} get np time = {thread_end_np_time - thread_start_np_time}")
            thread_start_saving_time = time.time()
            
            save_file_name = self.img_name.replace("jpg", "npy")
            np.save(f"{self.output_dir}/{save_file_name}", to_save_data_np)
            
            thread_end_saving_time = time.time()
            print(f"thread {self.idx} save np time saving time = {thread_end_saving_time - thread_start_saving_time}")
        else:
            img0_name = self.img_name.replace(".jpg", "_img0.png")
            img1_name = self.img_name.replace(".jpg", "_img1.png")
            img2_name = self.img_name.replace(".jpg", "_img2.png")
            cv2.imwrite(f"{self.output_dir}/{img0_name}", self.img0[self.idx, :, :h, :w].permute(1, 2, 0).cpu().numpy())
            cv2.imwrite(f"{self.output_dir}/{img1_name}", self.img1[self.idx, :, :h, :w].permute(1, 2, 0).cpu().numpy())
            cv2.imwrite(f"{self.output_dir}/{img2_name}", self.img2[self.idx, :, :h, :w].permute(1, 2, 0).cpu().numpy())


        # del self.img0, self.img1, self.img2
        # del self.img0_depth, self.img1_depth, self.img2_depth
        # del self.concat_img
        # del self.flow01, self.flow12, self.flow02
        # del self.back_flow01, self.back_flow12, self.back_flow02
        del self.to_save_data
        del to_save_data_np
        if not save_visual_data:
            del save_file_name
        del thread_start_saving_time, thread_end_saving_time
        del h, w, save_visual_data
        return


def multithread_get_img0_and_img0_depth_and_original_size(batch_size, img_names):
    # get imgs
    imgs_and_depths = [[0, 0] for _ in range(batch_size)]
    original_size = [[0, 0] for _ in range(batch_size)]
    threads = []
    for idx, img_name in enumerate(img_names):
        thread = MutliThreadLoad(idx, imgs_and_depths, original_size, img_name)
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    original_size = torch.tensor(original_size).to("cuda")
    
    # # find max size of imgs
    # max_h, max_w = 0, 0
    # for (h, w) in original_size: 
    #     max_h = max(max_h, h)
    #     max_w = max(max_w, w)

    # stack img and depth with max size(fill zero and fill 100)
    # img0 = torch.zeros((batch_size, 3, max_h, max_w)).type(torch.float32).to("cuda")
    # img0_depth = torch.ones((batch_size, 1, max_h, max_w)).type(torch.float32).to("cuda") * 100.
    # threads = []
    # for idx, (img, depth) in enumerate(imgs_and_depths):
    #     thread = MultiThreadFill(idx, img0, img0_depth, img, depth)
    #     threads.append(thread)
    #     thread.start()
    # for thread in threads:
    #     thread.join()

    # del imgs_and_depths
    # del thread, threads
    # del idx
    # del img, depth

    return img0, img0_depth, original_size, (max_h, max_w)


# def multithread_save_img_and_img_depth_and_flow(
#         img_names, original_size, output_dir, img0, img1, img2, img0_depth,
#         img1_depth, img2_depth, concat_img, flow01, flow12, flow02,
#         back_flow01, back_flow12, back_flow02):
    
#     threads = []
#     for idx, (img_name, (h, w)) in enumerate(zip(img_names, original_size)):
#         thread = MultiThreadSave(idx, img_name, (h, w), output_dir, img0, img1, img2, img0_depth,
#                                  img1_depth, img2_depth, concat_img, flow01, flow12, flow02,
#                                  back_flow01, back_flow12, back_flow02)
#         threads.append(thread)
#         thread.start()

#     for thread in threads:
#         thread.join()
    
#     del thread, threads
#     del img_name, h, w

# def multithread_save_data(img_names, original_size, output_dir, to_save_data):
#     threads = []
#     # for idx, (img_name, (h, w)) in enumerate(zip(img_names, original_size)):
#     for idx, img_name in enumerate(img_names):
#         thread = MultiThreadSave(idx, img_name, (h, w), output_dir, to_save_data)
#         threads.append(thread)
#         thread.start()

#     for thread in threads:
#         thread.join()
    
#     del thread, threads
#     del img_name, h, w


def batch_save_data(batch_idx, output_dir, to_save_data):
    get_np_start_time = time.time()
    
    to_save_data_np = to_save_data.detach().cpu().numpy()

    get_np_end_time = time.time()
    print(f"    get np time = {get_np_end_time - get_np_start_time}")
    save_np_start_time = time.time()

    np.savez_compressed(f"{output_dir}/{batch_idx}.npz",
            img_depth_flow=to_save_data_np)

    # np.savez_compressed(f"{output_dir}/{batch_idx}.npz",
    #         img_depth_flow=to_save_data_np,
    #         original_size=original_size_np)

    save_np_end_time = time.time()
    print(f"    save np time = {save_np_end_time - save_np_start_time}")

    del to_save_data_np
    # del original_size_np
    del get_np_start_time, get_np_end_time
    del save_np_start_time, save_np_end_time

