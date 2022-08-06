import numpy as np
from ctypes import cdll, c_void_p, c_int
import ctypes

# lib = cdll.LoadLibrary("lib/libwarping.so")
# warp = lib.forward_warping

def forward_warping(img, p1, z1, size):
    # h, w = size
    # safe_y = np.maximum(np.minimum(p1[:, :, :, 1].long(), h - 1), 0)
    # safe_x = np.maximum(np.minimum(p1[:, :, :, 0].long(), w - 1), 0)
    # warped_arr = np.zeros(h * w * 5).astype(np.uint8)
    # img = img.reshape(-1)
    # img = img.reshape(-1).astype(np.uint16)

    # warp(c_void_p(img.numpy().ctypes.data),
    #      c_void_p(safe_x[0].numpy().ctypes.data),
    #      c_void_p(safe_y[0].numpy().ctypes.data),
    #      c_void_p(z1.reshape(-1).numpy().ctypes.data),
    #      c_void_p(warped_arr.ctypes.data),
    #      c_int(h),
    #      c_int(w))
    # warped_arr = warped_arr.reshape(1, h, w, 5).astype(np.uint8)
    # return warped_arr
    
    h, w = size
    safe_y = np.maximum(np.minimum(p1[:, :, :, 1].long(), h - 1), 0)
    safe_x = np.maximum(np.minimum(p1[:, :, :, 0].long(), w - 1), 0)
    img = img.reshape((1, h, w, 3))
    warped = np.zeros((h, w, 5))
    dlut = np.ones((h, w)) * 1000
    for i in range(h):
        for j in range(w):
            x = safe_x[0, i, j]
            y = safe_y[0, i, j]
            if z1[0, i, j] < dlut[y, x]:
                for c in range(3):
                    warped[y, x, c] = img[0, i, j, c]

            warped[y, x, 3] = 1 
            if dlut[y, x] != 1000:
                warped[y, x, 4] = 0
            else:
                warped[y, x, 4] = 1
            dlut[y, x] = z1[0, i, j]

    warped = warped.reshape(1, h, w, 5)
    # .astype(np.uint8)
    return warped

import torch.nn as nn
import torch.nn.functional as F
import cv2
class Resample2d(nn.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        self.kernel_size = kernel_size
        # self.lib = cdll.LoadLibrary("lib/libwarping.so")
        self.lib = ctypes.CDLL("lib/libwarping.so")

    def forward(self, img, flow, depth, split=10):
        flow = flow[0].numpy()

        # input1_c  = input1.contiguous()
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_double_p = ctypes.POINTER(ctypes.c_double)

        if img.dtype == "uint8":
            img = img.astype("int32")

        if img.dtype == "int32":
            img_ctype_p = c_int_p
            c_func = self.lib.forward_warping_int
            print("int")
        elif img.dtype == "float32":
            img_ctype_p = c_float_p
            c_func = self.lib.forward_warping_float
            print("float")
        elif img.dtype == "float64":
            img_ctype_p = c_double_p
            c_func = self.lib.forward_warping_double
            print("double")

        print(f"{img.dtype = }")
        print(f"{flow.dtype = }")
        print(f"{img.shape = }")
        print(f"{flow.shape = }")

        _, _, c = img.shape
        h, w, _ = flow.shape

        sorted_depth = np.sort(depth[depth != 100])

        l = sorted_depth.shape[0]
        midpoints = [sorted_depth[i] for i in range(0, l, l // split)] + [sorted_depth[l - 1]]
        print(f"{midpoints = }")

        res = np.zeros_like(img)
        for i in range(split, 0, -1):
            # img = np.ascontiguousarray(img)
            flow = np.ascontiguousarray(flow)

            depth_cond = ((depth < midpoints[i]) & (depth >= midpoints[i - 1]))
            img_tmp = np.ascontiguousarray(img * np.stack((depth_cond, depth_cond, depth_cond), -1))
            # img_p = img_tmp.ctypes.data_as(c_int_p)
            img_p = img_tmp.ctypes.data_as(img_ctype_p)
            flow_p = flow.ctypes.data_as(c_float_p)
            output = np.zeros_like(img)
            print(f"{output.dtype = }")
            output = np.ascontiguousarray(output)
            output_p = output.ctypes.data_as(img_ctype_p)
            c_func(img_p, flow_p, output_p, c_int(self.kernel_size),
                                     c_int(h), c_int(w), c_int(c))

            # print(f"{np.min(output) = }")
            # print(f"{np.max(output) = }")

            res = res * (output == 0) + output
        return res


