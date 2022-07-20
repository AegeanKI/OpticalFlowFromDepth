import utils
import preprocess
import numpy as np
import math
import torch
# import torch.nn.functional as F
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import cv2
import matplotlib.pyplot as plt

#-----warping operation---CHR------
def occlude_det(xgrid, disp, if_flow_negative=True):
    #[batch, 1, H, W]     
    batch, chn, H, W = xgrid.size()
    print("width: ", W)
    
    xgrid = xgrid.detach().cpu().numpy()
    disp = disp.detach().cpu().numpy()
    
    flow_xgrid = xgrid + disp
    
#    accumulate_grid = flow_xgrid[:, :, :, 1:] - flow_xgrid[:, :, :, :W-1]
#    
#    accumulate_grid[accumulate <= 0] = 0
#    accumulate_grid[accumulate > 0] = 1
#    print(accumulate_grid.size())
    
    occlude_map = np.ones_like(flow_xgrid)
    
    for iter_batch in range(batch):
        for iter_chn in range(chn):
            for iter_row in range(H):
#                if(iter_row > 0):
#                    break
                
                # if(iter_row % 50 == 0):
                #     print("-----------processing %d row..---------" % (iter_row))
                idx_target = {}
                #table to store corresponding ref pixel coords for every idx of target coord
                for iter_col in range(W):
                    idx_target["%d" % (math.floor(flow_xgrid[iter_batch, iter_chn, iter_row, iter_col]))] = []
                    #initialize the table using flow_xgrid as keys
                    
                for iter_col in range(W):
                    #flow_xgrid has floating and negative values...
                    idx_target["%d" % (math.floor(flow_xgrid[iter_batch, iter_chn, iter_row, iter_col]))].append(iter_col)
                    #fill the table
#                    print("len table", len(idx_target))
                    
                    #for areas in ref image that map to areas outside of target image, they are valid too
                    #in short: set out-of-sight areas to zero
                    if not (0 < (math.floor(flow_xgrid[iter_batch, iter_chn, iter_row, iter_col])) < W):
                        occlude_map[iter_batch, iter_chn, iter_row, iter_col] = 0
                
                for iter_col in range(W):
                    if(if_flow_negative):
                        idx_target["%d" % (math.floor(flow_xgrid[iter_batch, iter_chn, iter_row, iter_col]))].pop()
                        #pop the max ref pixel coord,  which in this case (when flow is negative) means most front objects, then pixel coords left here are background objects being occluded in target image
                    else:
                        idx_target["%d" % (math.floor(flow_xgrid[iter_batch, iter_chn, iter_row, iter_col]))].pop(0)
#                    print(len(idx_target["%d" % (math.floor(flow_xgrid[iter_batch, iter_chn, iter_row, iter_col]))]))
                        #pop the min ref pixel coord, which in this case (when flow is positive) means most front objects, then pixel coords left here are background objects being occluded in target image
                    
                    
                    list_ref_idx = idx_target["%d" % (math.floor(flow_xgrid[iter_batch, iter_chn, iter_row, iter_col]))]
                    for iter_c, coord in enumerate(list_ref_idx):
                        #after popping out the front pixels, what ever left here are occluded pixels 
                        #set occluded areas to zero
                        occlude_map[iter_batch, iter_chn, iter_row, coord] = 0
                    
    
    print("number of occlude pixels: ",H * W - np.sum(occlude_map))
    
    return occlude_map

def apply_disparity(img, disp):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                            height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                            width, 1).transpose(1, 2).type_as(img)
    print(f"{x_base = }")

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                            # align_corners=False,
                            padding_mode='zeros')

    return output

if __name__ == "__main__":
    # img, h, w = utils.get_img("../im0.jpg")
    # depth = utils.get_depth("../d0.png", (h, w), 16)

    # vd = preprocess.VirtualDisparity(h, w)
    # disparity, stereo_img, stereo_depth = vd.forward(img, depth)
    # cv2.imwrite("output/stereo_img.png", stereo_img)
    # plt.imsave("output/stereo_depth_color.png", 1. / stereo_depth, cmap="magma")
    # cv2.imwrite("output/disparity.png", disparity * 20)

    # img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    # disparity = torch.from_numpy(disparity).unsqueeze(0).unsqueeze(0)
    # occlude_map = occlude_det(img, disparity)
    # occlude_map = torch.from_numpy(occlude_map[0]).permute(1, 2, 0).detach().numpy()
    # # print(f"{occlude_map = }")
    # img = img[0].permute(1, 2, 0).detach().numpy()
    # for i in range(h):
    #     for j in range(w):
    #         if np.any(occlude_map[i, j]):
    #             occlude_map[i, j] = (0, 0, 0)
    #         else:
    #             occlude_map[i, j] = (1, 1, 1)

    # cv2.imwrite("output/test.png", img * occlude_map)
    # cv2.imwrite("output/occlude_map.png", occlude_map * 100)

    # disparity = disparity / disparity.shape[3]
    # print(f"{torch.max(disparity) = }")
    # print(f"{torch.min(disparity) = }")
    # print(f"{disparity.shape = }")
    # output = apply_disparity(img.float(), disparity.float() / disparity.shape[3])
    # output = output[0].permute(1, 2, 0).detach().numpy()
    # cv2.imwrite("output/test.png", output)

    # acc = 0
    # img = img[0].permute(1, 2, 0).detach().numpy()
    # print(f"{img.shape = }")
    # print(f"{output.shape = }")
    # for i in range(3):
    #     for j in range(480):
    #         for k in range(640):
    #             acc = acc + (img[i, j, k] != output[i, j, k])
    # print(f"{acc = }")
    # print(f"{np.sum(img == output) = }")

    left = cv2.imread("../left.png", -1)
    right = cv2.imread("../right.png", -1)
    h, w, _ = left.shape

    disparity = cv2.imread("../disparity.png", -1)
    disparity = np.array(disparity, dtype=np.float32) / 256
    disparity = disparity / w

    # print(f"{np.min(disparity) = }")
    # print(f"{np.max(disparity) = }")

    # vd = preprocess.VirtualDisparity2(h, w)
    # depth, baseline = preprocess.VirtualDisparity2._convert_disparity_to_depth(disparity)

    # # depth = depth / np.max(depth)
    # # depth = 1 / depth
    # # depth = 1 / (depth + 0.005)
    # # depth[depth > 100] = 100

    # print(f"{np.min(depth) = }")
    # print(f"{np.max(depth) = }")
    # flow, img1, _ = vd.forward(left, depth, baseline)
    # plt.imsave("output/depth.png", 1 / depth, cmap="magma")
    # cv2.imwrite("output/test.png", img1)
    # # cv2.imwrite("output/test.png", warped_image[0])


    # img = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).float()
    # batch_size, _, height, width = img.size()
    # x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
    # y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)
    # disp = torch.from_numpy(disparity)
    # flow_field = torch.stack((x_base + disp, y_base), dim=3)
    # output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear', padding_mode='zeros')
    # img1 = output[0].permute(1, 2, 0).detach().numpy()
    # cv2.imwrite("output/test2.png", img1)

    def plot(imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()

    left = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).float()
    right = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0).float()

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    def preprocess(img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[368, 1240])
        img2_batch = F.resize(img2_batch, size=[368, 1240])
        return transforms(img1_batch, img2_batch)

    left, right = preprocess(left, right)

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
    model = model.eval()
    
    print(f"{left.shape = }")
    print(f"{right.shape = }")

    list_of_flows = model(left, right)
    predicted_flows = list_of_flows[-1]

    # flow = predicted_flows[0].permute(1, 2, 0).unsqueeze(0)
    # flow = torch.from_numpy(flow.detach().numpy())
    # .detach().numpy()
    # print(f"{flow.shape = }")
    from torchvision.utils import flow_to_image
    flow_imgs = flow_to_image(predicted_flows)
    grid = [[img1, flow_img] for (img1, flow_img) in zip(left, flow_imgs)]
    plot(grid)

    # flow_16bit, flow_color = utils.color_flow(flow)
    # cv2.imwrite("output/torch_model_flow_16bit.png", flow_16bit.astype(np.uint16))
    # cv2.imwrite("output/torch_model_flow_color.png", flow_color)
