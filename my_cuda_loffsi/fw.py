import math
from torch import nn
from torch.autograd import Function
import torch

import time
import fw_cuda

# torch.manual_seed(42)

class FW(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

    def set_shape(self, obj_shape):
        print(f"{obj_shape = }")

    def forward(self, obj, flow, depth):
        flow = flow.unsqueeze(0)
        obj = obj.unsqueeze(0)
        depth = depth.unsqueeze(0)
        # flow[((flow[:, 0:1] == 0) & (flow[:, 1:2] == 0)).repeat(1, 2, 1, 1)] = 1000

        _, _, h, w = obj.shape

        meshgrid = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        p0 = torch.stack(meshgrid, axis=0).type(torch.float32)
        p0 = p0.repeat(1, 1, 1, 1).to(self.device)

        p1 = p0 + flow
        # print(f"{p1.dtype = }")
        # print(f"{p0 = }")
        # print(f"{flow = }")
        # print(f"{p1 = }")

        safe_y = torch.clamp(p1[:, 1:2, ...], min=0, max=h-1)
        safe_x = torch.clamp(p1[:, 0:1, ...], min=0, max=w-1)
        
        obj = obj.contiguous().type(torch.float32)
        safe_y = safe_y.contiguous().type(torch.int64).type(torch.float32)
        safe_x = safe_x.contiguous().type(torch.int64).type(torch.float32)
        depth = depth.contiguous().type(torch.float32)
        
        # print(f"{obj.dtype = }")
        # print(f"{safe_y.dtype = }")
        # print(f"{safe_x.dtype = }")
        # print(f"{depth.dtype = }")
        # print(f"{same_range.dtype = }")

        output, valid, collision = fw_cuda.forward_warping(obj,
                                                    safe_y,
                                                    safe_x,
                                                    depth)

        output = output.to(self.device).type(torch.float32).squeeze(0)
        valid = valid.to(self.device).type(torch.float32).squeeze(0)
        collision = collision.to(self.device).type(torch.float32).squeeze(0)
        return output, valid, collision


