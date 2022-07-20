import numpy as np
from ctypes import cdll, c_void_p, c_int

lib = cdll.LoadLibrary("lib/libwarping.so")
warp = lib.forward_warping

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
