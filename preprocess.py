import utils
import using_clib
import geometry  
import flow_colors

import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import ctypes
import matplotlib.pyplot as plt
from bilateral_filter import sparse_bilateral_filtering

class VirtualEgoMotion():
    def __init__(self, h, w, no_sharp=False, another_K=False):
        self.h = h
        self.w = w
        self.no_sharp = no_sharp
        self.another_K = another_K

        self.K, self.inv_K = VirtualEgoMotion._create_plausible_K(another_K, h, w)
        self.p0 = VirtualEgoMotion._get_p0(h, w)

        self.backproject_depth = geometry.BackprojectDepth(1, h, w)
        self.project_3d = geometry.Project3D(1, h, w)

    @staticmethod
    def _create_plausible_K(another_K, h, w):
        K = np.array([[[0.58,    0, 0.5, 0],
                       # [0,    1.92, 0.5, 0],
                       [0,    0.58, 0.5, 0],
                       [0,       0,   1, 0], 
                       [0,       0,   0, 1]]], dtype=np.float32)

        if another_K:
            K[:, :2, :2] *= 2

        K[:, 0, :] *= w
        K[:, 1, :] *= h
        return torch.from_numpy(K), torch.from_numpy(np.linalg.pinv(K))
    
    @staticmethod
    def _get_p0(h, w):
        meshgrid = np.meshgrid(range(w), range(h), indexing="xy")
        p0 = np.stack(meshgrid, axis=-1).astype(np.float32)   
        return p0

    @staticmethod
    def _create_random_motion(axisangle_range, axisangle_base, translation_range, translation_base,
                              another_axisangle=None, another_translation=None):
        ax = utils.get_random(math.pi * axisangle_range, math.pi * axisangle_base)
        ay = utils.get_random(math.pi * axisangle_range, math.pi * axisangle_base)
        az = utils.get_random(math.pi * axisangle_range, math.pi * axisangle_base)
        camera_ang = [ax, ay, az]

        cx = utils.get_random(translation_range, translation_base)
        cy = utils.get_random(translation_range, translation_base)
        cz = utils.get_random(translation_range, translation_base)
        camera_mot = [cx, cy, cz]

        axisangle = torch.from_numpy(np.array([[camera_ang]], dtype=np.float32))
        translation = torch.from_numpy(np.array([[camera_mot]]))

        if another_axisangle is not None and another_translation is not None:
            T = geometry.transformation_from_parameters(axisangle + another_axisangle,
                                                        translation + another_translation)
        else:
            T = geometry.transformation_from_parameters(axisangle, translation)
        return T, axisangle, translation

    def forward(self, img, depth, segment=None):
        """
            input: img(HWC), depth(HW)
            output: flow_stereo(HWC), stereo_img_raw(HWC)
        """
        if not self.no_sharp:
            depth = sparse_bilateral_filtering(depth.copy(), img.copy(),
                                               filter_size=[5, 5], num_iter=2)

        img = torch.from_numpy(np.expand_dims(img, 0))
        # img = torch.from_numpy(np.expand_dims(img, 0)).to(torch.int16)
        depth = torch.from_numpy(np.expand_dims(depth, 0)).float()
        cam_points = self.backproject_depth(depth, self.inv_K)

        T1, axisangle, translation = VirtualEgoMotion._create_random_motion(1. / 36., 1. / 36.,
                                                                            0.1, 0.1) 
        # T1, axisangle, translation = VirtualEgoMotion._create_random_motion(1. / 72., 1. / 72.,
        #                                                                     0.05, 0.05) 
        p1, z1 = self.project_3d(cam_points, self.K, T1)
        z1 = z1.reshape(1, self.h, self.w)
        
        if segment is not None:
            instances, labels = segment
            for l in range(len(labels)):
                Ti, _, _ = VirtualEgoMotion._create_random_motion(1. / 72., 1. / 72., 0.05, 0.05,
                                                                  axisangle, translation)
                pi, zi = self.project_3d(cam_points, self.K, Ti)
                zi = zi.reshape(1, self.h, self.w)
                p1[instances[l] > 0] = pi[instances[l] > 0]
                z1[instances[l][:, :, :, 0] > 0] = zi[instances[l][:, :, :, 0] > 0]


        p1 = (p1 + 1) / 2
        p1[:, :, :, 0] *= self.w - 1
        p1[:, :, :, 1] *= self.h - 1

        warped_img = using_clib.forward_warping(img, p1, z1, (self.h, self.w)) 

        flow_stereo = p1 - self.p0
        stereo_img_raw = warped_img[0, :, :, 0:3]
        
        masks = {}
        masks["H"] = warped_img[0 ,:, :, 3:4]
        masks["M"] = warped_img[0, :, :, 4:5]
        masks["M"] = 1 - (masks["M"] == masks["H"]).astype(np.uint8)
        masks["M'"] = cv2.dilate(masks["M"], np.ones((3, 3), np.uint8), iterations=1)
        masks["P"] = (np.expand_dims(masks["M'"], -1) == masks["M"]).astype(np.uint8)
        masks["H'"] = masks["H"] * masks["P"]

        return flow_stereo, stereo_img_raw, masks

# class VirtualDisparity():
#     def __init__(self, h, w, no_sharp=False):
#         self.h = h
#         self.w = w

#         self.xs, self.ys = VirtualDisparity._get_xs_ys(h, w) 

#     @staticmethod
#     def _get_xs_ys(h, w):
#         xs, ys = np.meshgrid(range(w), range(h))
#         return torch.from_numpy(xs).float(), torch.from_numpy(ys).float()

#     @staticmethod
#     def _convert_depth_to_disparity(depth):
#         s = utils.get_random(175, 50, random_sign=False) / depth.shape[1]
        
#         # print(f"{s = }")
#         depth_max = np.max(depth)
#         # print(f"{depth_max = }")
#         disparity = (1. / depth) * s * depth_max
#         return disparity

#     def forward(self, img, depth):
#         """
#             input: img(HWC), depth(HW)
#             output: disparity(HW), res(HWC)
#         """
#         disparity = VirtualDisparity._convert_depth_to_disparity(depth)
#         img = torch.from_numpy(img).permute(2, 0, 1)
#         depth = torch.from_numpy(depth)
        
#         new_xs = self.xs - disparity
#         new_xs = ((new_xs / (self.w - 1)) - 0.5) * 2
#         new_ys = ((self.ys / (self.h - 1)) - 0.5) * 2
#         sample_pix = torch.stack([new_xs, new_ys], 2)
        
#         warped_img = F.grid_sample(img.unsqueeze(0).float(), sample_pix.unsqueeze(0).float(),
#                                    padding_mode='zeros', align_corners=False)
#         warped_depth = F.grid_sample(depth.unsqueeze(0).unsqueeze(0).float(), sample_pix.unsqueeze(0).float(),
#                                      padding_mode='zeros', align_corners=False)

#         stereo_img = warped_img[0].permute(1, 2, 0)
#         stereo_depth = warped_depth[0, 0]
#         stereo_depth[(self.xs - disparity) < 0] = 100

#         # disparity, stereo_img = disparity.detach().numpy(), stereo_img.detach().numpy()
#         stereo_img = stereo_img.detach().numpy()
#         stereo_depth = stereo_depth.detach().numpy()
#         # stereo_img = stereo_img_raw
#         return disparity, stereo_img, stereo_depth

class VirtualDisparity2(VirtualEgoMotion):
    def __init__(self, h, w, no_sharp=False):
        super().__init__(h, w, no_sharp)

    @staticmethod
    def _create_plausible_f():
        # return 0.5 * 1.3
        return 1

    @staticmethod
    def _create_plausible_baseline():
        # return 0.1 / 1.3
        return 50
        B = utils.get_random(0.05, 0.05, random_sign=False)
        return B

    def forward(self, img, depth, baseline=None, depth_raw=None):
        """
            input: img(HWC), depth(HW)
            output: flow_stereo(HWC), stereo_img_raw(HWC)
        """
        if not self.no_sharp:
            depth = sparse_bilateral_filtering(depth.copy(), img.copy(),
                                               filter_size=[5, 5], num_iter=2)

        img = torch.from_numpy(np.expand_dims(img, 0))
        # img = torch.from_numpy(np.expand_dims(img, 0)).to(torch.int32)
        depth = torch.from_numpy(np.expand_dims(depth, 0)).float()
        cam_points = self.backproject_depth(depth, self.inv_K)
        if baseline is None:
            baseline = VirtualDisparity2._create_plausible_baseline()
        baseline = baseline / self.w
        cam_points[:, 0, ...] = cam_points[:, 0, ...] - baseline 

        T1 = torch.eye(4).unsqueeze(0)

        p1, z1 = self.project_3d(cam_points, self.K, T1)
        z1 = z1.reshape(1, self.h, self.w)
        
        p1 = (p1 + 1) / 2
        p1[:, :, :, 0] *= self.w - 1
        p1[:, :, :, 1] *= self.h - 1

        # print(f"{self.p0 = }")
        # print(f"{p1 = }")
        # print(f"{self.p0.shape = }")
        # print(f"{p1.shape = }")


        warped_img = using_clib.forward_warping(img, p1, z1, (self.h, self.w)) 

        flow_stereo = p1 - self.p0
        stereo_img_raw = warped_img[0, :, :, 0:3]

        masks = {}
        masks["H"] = warped_img[0 ,:, :, 3:4]
        masks["M"] = warped_img[0, :, :, 4:5]
        masks["M"] = 1 - (masks["M"] == masks["H"]).astype(np.uint8)
        masks["M'"] = cv2.dilate(masks["M"], np.ones((3, 3), np.uint8), iterations=1)
        masks["P"] = (np.expand_dims(masks["M'"], -1) == masks["M"]).astype(np.uint8)
        masks["H'"] = masks["H"] * masks["P"]

        if depth_raw is not None:
            depth_raw = depth_raw.astype(np.int32)
            depth_raw = torch.from_numpy(depth_raw)
            depth = torch.stack((depth_raw, depth_raw, depth_raw), -1)
        else:
            depth = torch.stack((depth, depth, depth), -1)
        warped_depth = using_clib.forward_warping(depth, p1, z1, (self.h, self.w)) 
        stereo_depth_raw = warped_depth[0, :, :, 0] 
        # / 255

        return flow_stereo, stereo_img_raw, masks, stereo_depth_raw

    @staticmethod
    def _convert_disparity_to_depth(disparity):
        baseline = VirtualDisparity2._create_plausible_baseline()
        f = VirtualDisparity2._create_plausible_f()
        depth = baseline * f / (disparity + 0.005)
        depth = (np.max(depth) - depth) / (np.max(depth) - np.min(depth))
        depth = utils.normalize_depth(depth)
        return depth, baseline


if __name__ == "__main__":
    img, h, w = utils.get_img("../im0.jpg") # CHW
    depth, depth_raw = utils.get_depth("../d0.png", (h, w), 16) # HW
    # print(f"{np.min(depth_raw) = }")
    # print(f"{np.max(depth_raw) = }")
    # depth = depth_raw / (2 ** 16 - 1)
    # depth = utils.normalize_depth(depth)
    instances_mask, instances, labels = utils.get_mask("../s0.png", (h, w)) # HW

    vem = VirtualEgoMotion(h, w)
    # flow_stereo, stereo_img_raw, masks = vem.forward(img, depth, (instances, labels))
    flow_stereo, stereo_img_raw, masks = vem.forward(img, depth)

    print(f"{flow_stereo.shape = }")
    print(f"{flow_stereo.dtype = }")
    print(f"{torch.min(flow_stereo) = }")
    print(f"{torch.max(flow_stereo) = }")
    
    flow_16bit, flow_color = utils.color_flow(flow_stereo)
    # stereo_img = utils.inpaint_img(stereo_img_raw, masks)

    cv2.imwrite("output/original_img.png", img)
    cv2.imwrite("output/stereo_img_raw.png", stereo_img_raw)
    # cv2.imwrite("output/stereo_img.png", stereo_img)
    cv2.imwrite("output/flow_16bit.png", flow_16bit.astype(np.uint16))
    cv2.imwrite("output/flow_color.png", flow_color)
    cv2.imwrite("output/H.png", masks["H"] * 255)
    cv2.imwrite("output/M.png", masks["M"] * 255)
    cv2.imwrite("output/P.png", masks["P"] * 255)
    plt.imsave("output/depth_color.png", 1. / depth, cmap="magma")
    plt.imsave("output/instance_color.png", instances_mask, cmap="magma")

    # vd = VirtualDisparity(h, w)
    vd = VirtualDisparity2(h, w)
    disparity, stereo_img, masks, stereo_depth = vd.forward(img, depth, None, depth_raw)

    # print(f"{depth_raw.shape = }")
    stereo_depth = utils.normalize_depth(stereo_depth)

    cv2.imwrite("output/stereo_img.png", stereo_img)
    plt.imsave("output/stereo_depth_color.png", 1. / stereo_depth, cmap="magma")
    # cv2.imwrite("output/disparity.png", disparity * 20)
    

    left, right, h, w = utils.get_stereo_img("../left.png", "../right.png")

    disparity = utils.get_disparity("../disparity.png", (h, w))


    test = np.zeros((1, h, w, 2))
    test[0, :, :, 0] = -disparity * w
    test = torch.from_numpy(test).float()
    print(f"{test.shape = }")
    print(f"{test.dtype = }")
    print(f"{torch.min(test) = }")
    print(f"{torch.max(test) = }")
    flow_16bit, flow_color = utils.color_flow(test)
    cv2.imwrite("output/test_flow_16bit.png", flow_16bit.astype(np.uint16))
    cv2.imwrite("output/test_flow_color.png", flow_color)


    vd = VirtualDisparity2(h, w)
    depth, baseline = VirtualDisparity2._convert_disparity_to_depth(disparity)
    

    flow_stereo, stereo_img_raw, masks, stereo_depth = vd.forward(left, depth, baseline, None)
    print(f"{torch.min(flow_stereo) = }")
    print(f"{torch.max(flow_stereo) = }")
    flow_16bit, flow_color = utils.color_flow(flow_stereo)
    # stereo_img = utils.inpaint_img(stereo_img_raw, masks)
    cv2.imwrite("output/original_left.png", left)
    cv2.imwrite("output/original_right.png", right)
    cv2.imwrite("output/disparity_to_flow_16bit.png", flow_16bit.astype(np.uint16))
    cv2.imwrite("output/disparity_to_flow_color.png", flow_color)
    plt.imsave("output/disparity_to_depth.png", 1 / depth, cmap="magma")
    cv2.imwrite("output/disparity_warp_raw.png", stereo_img_raw)
    # cv2.imwrite("output/disparity_warp.png", stereo_img)

    ctypes._reset_cache()


