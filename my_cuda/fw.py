import math
from torch import nn
from torch.autograd import Function
import torch

import fw_cuda

# torch.manual_seed(42)

class FW(nn.Module):
    def __init__(self, size, batch_size, device="cuda"):
        super().__init__()
        self.h, self.w = size
        self.device = device

        meshgrid = torch.meshgrid(torch.arange(self.w), torch.arange(self.h), indexing="xy")
        self.p0 = torch.stack(meshgrid, axis=0).type(torch.float32)
        self.p0 = self.p0.repeat(batch_size, 1, 1, 1).to(device)

    def forward(self, obj, flow, depth, same_range=0.2):
        # flow = flow.contiguous().type(torch.float32)
        # print(f"{flow = }")
        flow[((flow[:, 0:1] == 0) & (flow[:, 1:2] == 0)).repeat(1, 2, 1, 1)] = 1000

        p1 = self.p0 + flow

        safe_y = torch.clamp(p1[:, 1:2, ...], min=0, max=self.h - 1)
        safe_x = torch.clamp(p1[:, 0:1, ...], min=0, max=self.w - 1)
        
        obj = obj.contiguous().type(torch.float32)
        safe_y = safe_y.contiguous().type(torch.float32)
        safe_x = safe_x.contiguous().type(torch.float32)
        depth = depth.contiguous().type(torch.float32)

        # print(f"{obj.dtype = }")
        # print(f"{safe_y.dtype = }")
        # print(f"{safe_x.dtype = }")
        # print(f"{depth.dtype = }")
        # print(f"{same_range.dtype = }")

        output, = fw_cuda.forward_warping(
                obj,
                safe_y,
                safe_x,
                depth,
                same_range)
        del obj, safe_y, safe_x, depth
        return output.to(self.device).type(torch.float32)


