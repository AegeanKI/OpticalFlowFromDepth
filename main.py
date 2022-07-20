import torch
# from utils import get_rgb, get_depth, get_plausible_k
import utils
import using_clib
import geometry  
import flow_colors
import numpy as np
import cv2
import ctypes
import matplotlib.pyplot as plt
from bilateral_filter import sparse_bilateral_filtering

if __name__ == "__main__":
    rgb, h, w = utils.get_rgb("../im0.jpg")
    depth = utils.get_depth("../d0.png", (h, w), 16)

    depth = sparse_bilateral_filtering( depth.copy(), rgb.copy(), filter_size=[5, 5], num_iter=2, )

    K, inv_K = utils.get_plausible_K(h, w)

    rgb = torch.from_numpy(np.expand_dims(rgb, 0))
    depth = torch.from_numpy(np.expand_dims(depth, 0)).float()
    K = torch.from_numpy(K)
    inv_K = torch.from_numpy(inv_K)

    backproject_depth = geometry.BackprojectDepth(1, h, w)
    project_3d = geometry.Project3D(1, h, w)

    p0 = utils.get_p0(h, w)

    instances_mask, instances, labels = utils.get_mask("../s0.png", (h, w))
    cam_points = backproject_depth(depth, inv_K)

    num_motions = 3
    for idm in range(num_motions):
        T1 = utils.get_random_motion(0.1, 0.1, 1. / 36., 1. / 36.) 
        p1, z1 = project_3d(cam_points, K, T1)
        z1 = z1.reshape(1, h, w)
        
        for l in range(len(labels)):
            Ti = utils.get_random_motion(0.05, 0.05, 1. / 72., 1. / 72.)
            pi, zi = project_3d(cam_points, K, Ti)
            p1[instances[l] > 0] = pi[instances[l] > 0]
            zi = zi.reshape(1, h, w)
            z1[instances[l][:, :, :, 0] > 0] = zi[instances[l][:, :, :, 0] > 0]


        p1 = (p1 + 1) / 2
        p1[:, :, :, 0] *= w - 1
        p1[:, :, :, 1] *= h - 1

        warped_arr = using_clib.forward_warping(rgb, p1, z1, (h, w)) 

        flow_01 = p1 - p0
        flow_16bit = cv2.cvtColor( np.concatenate((flow_01 * 64. + (2**15), np.ones_like(flow_01)[:,:,:,0:1]), -1)[0], cv2.COLOR_BGR2RGB )
        flow_color = flow_colors.flow_to_color(flow_01[0].numpy(), convert_to_bgr=True)


        im1_raw = warped_arr[0,:,:,0:3]
        im1 = cv2.inpaint(im1_raw, 1 - warped_arr[0,:,:,3:4], 3, cv2.INPAINT_TELEA)

        cv2.imwrite("output/my_im0.png", rgb[0].numpy())
        cv2.imwrite("output/my_im1_raw.png", im1_raw)
        cv2.imwrite("output/my_im1.png", im1)
        cv2.imwrite("output/my_flow_16bit.png", flow_16bit.astype(np.uint16))
        cv2.imwrite("output/my_flow_color.png", flow_color)
        plt.imsave("output/my_depth_color.png", 1. / depth[0].detach().numpy(), cmap="magma")
        plt.imsave("output/my_instance_color.png", instances_mask, cmap="magma")


        ctypes._reset_cache()


