import math
from torch import nn
from torch.autograd import Function
import torch

import fw_cuda

torch.manual_seed(42)

class FW(nn.Module):
    def __init__(self, size, batch_size):
        super().__init__()
        self.h, self.w = size

        meshgrid = torch.meshgrid(torch.arange(self.w), torch.arange(self.h), indexing="xy")
        self.p0 = torch.stack(meshgrid, axis=0).type(torch.float64)
        self.p0 = self.p0.repeat(batch_size, 1, 1, 1).to("cuda")

    def forward(self, obj, flow, depth, same_range=0.2):
        p1 = self.p0 + flow

        safe_y = torch.clamp(p1[:, 1:2, ...], min=0, max=self.h - 1)
        safe_x = torch.clamp(p1[:, 0:1, ...], min=0, max=self.w - 1)
        
        obj = obj.contiguous()
        safe_y = safe_y.contiguous()
        safe_x = safe_x.contiguous()
        depth = depth.contiguous()

        output, = fw_cuda.forward_warping(
                obj,
                safe_y,
                safe_x,
                depth,
                same_range)
        return output


