import numpy as np
from ctypes import cdll, c_void_p, c_int, c_double
import ctypes
import pycuda.autoinit
from pycuda.compiler import SourceModule


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
        c_double_p = ctypes.POINTER(ctypes.c_double)

        depth_p = depth.ctypes.data_as(c_double_p)
        output = np.zeros_like(depth)
        output_p = output.ctypes.data_as(c_double_p)

        self.lib.warp_depth(safe_y_p, safe_x_p, depth_p,
                            c_int(h), c_int(w), c_double(same_range), output_p)
        return output

    def warp_flow(self, flow, safe_y_p, safe_x_p, depth, h, w, same_range):
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

    def forward(self, img, flow, depth, size, same_range=0.2):
        h, w = size
        meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
        p0 = np.stack(meshgrid, axis=-1).astype(np.float64)
        p0 = torch.from_numpy(p0).unsqueeze(0)
        p1 = p0 + flow
        safe_y = np.maximum(np.minimum(p1[:, :, :, 1], h - 1), 0)
        safe_x = np.maximum(np.minimum(p1[:, :, :, 0], w - 1), 0)
        safe_y = np.ascontiguousarray(safe_y)
        safe_x = np.ascontiguousarray(safe_x)
        
        c_double_p = ctypes.POINTER(ctypes.c_double)
        safe_y_p = safe_y[0].ctypes.data_as(c_double_p) 
        safe_x_p = safe_x[0].ctypes.data_as(c_double_p)


        if img.ndim == 2:
            return self.warp_depth(safe_y_p, safe_x_p, depth, h, w, same_range)
        if img.shape[2] == 2:
            img = torch.from_numpy(img).unsqueeze(0) # it's a flow
            return self.warp_flow(img, safe_y_p, safe_x_p, depth, h, w, same_range)
        return self.warp_img(img, safe_y_p, safe_x_p, depth, h, w, same_range)


    def warp_img(self, img, safe_y_p, safe_x_p, depth, h, w, same_range):
        c_double_p = ctypes.POINTER(ctypes.c_double)

        img = img.astype("float64")
        img = np.ascontiguousarray(img)
        img_p = img.ctypes.data_as(c_double_p)
        depth = np.ascontiguousarray(depth)
        depth_p = depth.ctypes.data_as(c_double_p)
        output = np.zeros_like(img)
        output = np.ascontiguousarray(output)
        output_p = output.ctypes.data_as(c_double_p)

        self.lib.forward_warping_img(img_p, safe_y_p, safe_x_p, depth_p,
                                     c_int(h), c_int(w), c_double(same_range), output_p)
        return output

    def warp_depth(self, safe_y_p, safe_x_p, depth, h, w, same_range):
        c_double_p = ctypes.POINTER(ctypes.c_double)

        depth = np.ascontiguousarray(depth)
        depth_p = depth.ctypes.data_as(c_double_p)
        output = np.zeros_like(depth)
        output = np.ascontiguousarray(output)
        output_p = output.ctypes.data_as(c_double_p)

        self.lib.forward_warping_depth(safe_y_p, safe_x_p, depth_p,
                            c_int(h), c_int(w), c_double(same_range), output_p)
        return output

    def warp_flow(self, flow, safe_y_p, safe_x_p, depth, h, w, same_range):
        c_double_p = ctypes.POINTER(ctypes.c_double)
        
        flow = flow[0].numpy()
        flow = np.ascontiguousarray(flow)
        flow_p = flow.ctypes.data_as(c_double_p)
        depth = np.ascontiguousarray(depth)
        depth_p = depth.ctypes.data_as(c_double_p)
        output = np.zeros_like(flow)
        output = np.ascontiguousarray(output)
        output_p = output.ctypes.data_as(c_double_p)

        self.lib.forward_warping_flow(flow_p, safe_y_p, safe_x_p, depth_p,
                                      c_int(h), c_int(w), c_double(same_range), output_p)

        return output

class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super().__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer():
        return self.t.data_ptr()

class Resample2d2(nn.Module):
    def __init__(self, size, batch_size, same_range=0.2):
        super().__init__()
        self.size = size
        self.h, self.w = size
        self.batch_size = batch_size
        self.same_range = same_range

        meshgrid = torch.meshgrid(torch.arange(self.w), torch.arange(self.h), indexing="xy")
        self.p0 = torch.stack(meshgrid, axis=0).type(torch.float64)
        self.p0 = self.p0.repeat(batch_size, 1, 1, 1).to("cuda")

        self.lib = ctypes.CDLL("lib/libwarping.so")
        
        with open("testpycudafunction.cu") as f:
            kernel_code = f.read()
        mod = SourceModule(kernel_code, no_extern_c=True)
        self.forward_warping_flow = mod.get_function("forward_warping_flow")

    def _contiguous(self, arr):
        return arr if arr.is_contiguous() else arr.contiguous()

    def forward(self, obj, flow, depth):
        p1 = self.p0 + flow

        safe_y = torch.clamp(p1[:, 1:2, ...], min=0, max=self.h - 1)
        safe_x = torch.clamp(p1[:, 0:1, ...], min=0, max=self.w - 1)
        
        # if obj.shape[1] == 1: # obj is depth
        #     pass
        # elif obj.shape[1] == 2: # obj is flow
        #     return self.warp_flow(obj, safe_y, safe_x, depth)
        # elif obj.shape[1] == 3: # obj is img
        #     pass
        # return

        output = torch.zeros_like(obj).to("cuda")
        dlut = torch.ones_like(depth).to("cuda") * 100.
        dlut_count = torch.zeros_like(depth).to("cuda")
        c = obj.shape[1]

        obj = self._contiguous(obj)
        safe_y = self._contiguous(safe_y)
        safe_x = self._contiguous(safe_x)
        depth = self._contiguous(depth)
        output = self._contiguous(output)
        dlut = self._contiguous(dlut)
        dlut_count = self._contiguous(dlut_count)

        self.forward_warping_flow(Holder(obj), Holder(safe_y), Holder(safe_x), Holder(depth),
                                  np.int32(self.batch_size), np.int32(c), np.int32(self.h), np.int32(self.w),
                                  np.float64(self.same_range), Holder(output), Holder(dlut), Holder(dlut_count),
                                  block=(1, 1, 1), grid=(1, 1))
        return output

        # if img.ndim == 2:
        #     return self.warp_depth(safe_y_p, safe_x_p, depth, h, w, same_range)
        # if img.shape[2] == 2:
        #     img = torch.from_numpy(img).unsqueeze(0) # it's a flow
        #     return self.warp_flow(img, safe_y_p, safe_x_p, depth, h, w, same_range)
        # return self.warp_img(img, safe_y_p, safe_x_p, depth, h, w, same_range)


    def warp_img(self, img, safe_y_p, safe_x_p, depth, h, w, same_range):
        c_double_p = ctypes.POINTER(ctypes.c_double)

        img = img.astype("float64")
        img_p = img.ctypes.data_as(c_double_p)
        depth_p = depth.ctypes.data_as(c_double_p)
        output = np.zeros_like(img)
        output_p = output.ctypes.data_as(c_double_p)

        self.lib.forward_warping_img(img_p, safe_y_p, safe_x_p, depth_p,
                                     c_int(h), c_int(w), c_double(same_range), output_p)
        return output

    def warp_depth(self, safe_y_p, safe_x_p, depth, h, w, same_range):
        c_double_p = ctypes.POINTER(ctypes.c_double)

        depth_p = depth.ctypes.data_as(c_double_p)
        output = np.zeros_like(depth)
        output_p = output.ctypes.data_as(c_double_p)

        self.lib.forward_warping_depth(safe_y_p, safe_x_p, depth_p,
                            c_int(h), c_int(w), c_double(same_range), output_p)
        return output

    def warp_flow(self, obj, safe_y, safe_x, depth):
        output = torch.zeros_like(obj).to("cuda")
        dlut = torch.ones_like(depth).to("cuda") * 100.
        dlut_count = torch.zeros_like(depth).to("cuda")

        c = 2
        self.forward_warping_flow(Holder(obj), Holder(safe_y), Holder(safe_x), Holder(depth),
                                  np.int32(self.batch_size), np.int32(c), np.int32(self.h), np.int32(self.w),
                                  np.float64(self.same_range), Holder(output), Holder(dlut), Holder(dlut_count),
                                  block=(1, 1, 1), grid=(1, 1))
        return output

