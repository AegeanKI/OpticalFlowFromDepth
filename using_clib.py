import numpy as np
from ctypes import cdll, c_void_p, c_int, c_double
import ctypes

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class ForwardWarping(nn.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.lib = ctypes.CDLL("lib/libwarping.so")

    def forward(self, img, flow, depth, size, same_range=1.0):
        h, w = size
        meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
        p0 = np.stack(meshgrid, axis=-1).astype(np.float64)
        p0 = torch.from_numpy(p0).unsqueeze(0)
        p1 = p0 + flow
        safe_y = np.maximum(np.minimum(p1[:, :, :, 1], h - 1), 0)
        safe_x = np.maximum(np.minimum(p1[:, :, :, 0], w - 1), 0)
        
        c_double_p = ctypes.POINTER(ctypes.c_double)
        safe_y_p = safe_y[0].numpy().ctypes.data_as(c_double_p) 
        safe_x_p = safe_x[0].numpy().ctypes.data_as(c_double_p)


        if img.ndim == 2:
            return self.warp_depth(safe_y_p, safe_x_p, depth, h, w, same_range)
        if img.shape[2] == 2:
            img = torch.from_numpy(img).unsqueeze(0) # it's a flow
            return self.warp_flow(img, safe_y_p, safe_x_p, depth, h, w, same_range)
        return self.warp_img(img, safe_y_p, safe_x_p, depth, h, w, same_range)


    def warp_img(self, img, safe_y_p, safe_x_p, depth, h, w, same_range):
        print(f"warp img")
        c_double_p = ctypes.POINTER(ctypes.c_double)

        img = img.astype("float64")
        img_p = img.ctypes.data_as(c_double_p)
        depth_p = depth.ctypes.data_as(c_double_p)
        output = np.zeros_like(img)
        output_p = output.ctypes.data_as(c_double_p)

        self.lib.warp_img(img_p, safe_y_p, safe_x_p, depth_p,
                          c_int(h), c_int(w), c_double(same_range), output_p)
        return output

    def warp_depth(self, safe_y_p, safe_x_p, depth, h, w, same_range):
        print(f"warp depth")
        c_double_p = ctypes.POINTER(ctypes.c_double)

        depth_p = depth.ctypes.data_as(c_double_p)
        output = np.zeros_like(depth)
        output_p = output.ctypes.data_as(c_double_p)

        self.lib.warp_depth(safe_y_p, safe_x_p, depth_p,
                            c_int(h), c_int(w), c_double(same_range), output_p)
        return output

    def warp_flow(self, flow, safe_y_p, safe_x_p, depth, h, w, same_range):
        print("warp flow")
        c_double_p = ctypes.POINTER(ctypes.c_double)
        
        flow = flow[0].numpy()
        flow_p = flow.ctypes.data_as(c_double_p)
        depth_p = depth.ctypes.data_as(c_double_p)
        output = np.zeros_like(flow)
        output_p = output.ctypes.data_as(c_double_p)

        self.lib.warp_flow(flow_p, safe_y_p, safe_x_p, depth_p,
                            c_int(h), c_int(w), c_double(same_range), output_p)

        return output

class Resample2d(nn.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.lib = ctypes.CDLL("lib/libwarping.so")

    def forward(self, img, flow, depth, split=10, warp_depth=False):
        if warp_depth:
            return self.forward_depth(flow, depth, split)
        flow = flow[0].numpy()

        # input1_c  = input1.contiguous()
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_double_p = ctypes.POINTER(ctypes.c_double)

        if img.dtype == "uint8" or img.dtype == "bool":
            img = img.astype("int32")

        if img.dtype == "int32":
            img_ctype_p = c_int_p
            c_func = self.lib.forward_warping_int
            print("int")
        elif img.dtype == "float64":
            img_ctype_p = c_double_p
            c_func = self.lib.forward_warping_double
            print("double")

        _, _, c = img.shape
        h, w, _ = flow.shape

        flow = np.ascontiguousarray(flow)
        img_p = img.ctypes.data_as(img_ctype_p)
        flow_p = flow.ctypes.data_as(c_float_p)
        depth_p = depth.ctypes.data_as(c_double_p)
        output = np.zeros_like(img)
        output = np.ascontiguousarray(output)
        output_p = output.ctypes.data_as(img_ctype_p)
        c_func(img_p, flow_p, depth_p, output_p, c_int(self.kernel_size),
               c_int(h), c_int(w), c_int(c))

        return output

    def forward_depth(self, flow, depth, split=10):
        flow = flow[0].numpy()

        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_double_p = ctypes.POINTER(ctypes.c_double)

        c_func = self.lib.forward_warping_depth
        output = np.zeros_like(depth)

        h, w, _ = flow.shape

        flow = np.ascontiguousarray(flow)
        flow_p = flow.ctypes.data_as(c_float_p)
        depth_p = depth.ctypes.data_as(c_double_p)
        output = np.zeros_like(depth)
        output = np.ascontiguousarray(output)
        output_p = output.ctypes.data_as(c_double_p)
        c_func(flow_p, depth_p, output_p, c_int(self.kernel_size),
               c_int(h), c_int(w))

        return output

