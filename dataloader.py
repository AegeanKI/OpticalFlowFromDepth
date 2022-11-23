import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2

def getData(mode):
    if mode == 'train':
        # img = pd.read_csv('train_img.csv', header=None)
        # label = pd.read_csv('train_label.csv', header=None)
        fp = open('dCOCO_file_list.txt')
        img_dirs = fp.readlines()
        fp.close()

        img_dirs = [img_dir[:-1] for img_dir in img_dirs]
        return img_dirs
    else:
        # img = pd.read_csv('test_img.csv', header=None)
        # label = pd.read_csv('test_label.csv', header=None)
        # return np.squeeze(img.values), np.squeeze(label.values)
        return None


class dCOCODataset(data.Dataset):
    def __init__(self, mode, mono=True, device="cuda"):
        # self.root = root
        # self.img_name, self.label = getData(mode)
        # self.img_dirs = getData(mode)
        self.mode = mode
        self.mono = mono
        self.device = device
        # print(f"> Found {len(self.img_dirs)} dirs...")
        batch_size = 4
        self.every_batch_change_dir = 320 // batch_size

    def __len__(self):
        # return len(self.img_dirs)
        # return 20000 // 8
        # return 1371
        if self.mono:
            return 919
        else:
            return 97

    def __getitem__(self, batch_idx):
        if self.mono:
            batch_dir = f"output/monocular/{batch_idx // self.every_batch_change_dir}"
        else:
            batch_dir = f"output/stereo/{batch_idx // self.every_batch_change_dir}"

        npz_file = np.load(f"{batch_dir}/{batch_idx}.npz")
        img_depth_flow = npz_file['img_depth_flow']
        # return torch.from_numpy(img_depth_flow).to(self.device)
        return img_depth_flow

        # get_dir_path = lambda index: f"output/monocular/{self.img_dirs[index]}"
        # transform = transforms.Compose([transforms.ToTensor()])
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize([384, 384]),
        # ])
        # transform_flow = transforms.Compose([
        #     transforms.Resize([384, 384]),
        # ])

        # img0 = transform(np.load(f"{get_dir_path(index)}/img0.npy"))
        # img1 = transform(np.load(f"{get_dir_path(index)}/img1.npy"))
        # img2 = transform(np.load(f"{get_dir_path(index)}/img2.npy"))
        # img0_depth = transform(np.load(f"{get_dir_path(index)}/img0_depth.npy"))
        # img1_depth = transform(np.load(f"{get_dir_path(index)}/img1_depth.npy"))
        # img2_depth = transform(np.load(f"{get_dir_path(index)}/img2_depth.npy"))
        # flow01 = transform_flow(torch.load(f"{get_dir_path(index)}/flow01.pt").squeeze().permute(2, 0, 1))
        # flow12 = transform_flow(torch.load(f"{get_dir_path(index)}/flow12.pt").squeeze().permute(2, 0, 1))
        # flow02 = transform_flow(torch.load(f"{get_dir_path(index)}/flow02.pt").squeeze().permute(2, 0, 1))
        # back_flow01 = transform_flow(torch.load(f"{get_dir_path(index)}/back_flow01.pt").squeeze().permute(2, 0, 1))
        # back_flow12 = transform_flow(torch.load(f"{get_dir_path(index)}/back_flow12.pt").squeeze().permute(2, 0, 1))
        # back_flow02 = transform_flow(torch.load(f"{get_dir_path(index)}/back_flow02.pt").squeeze().permute(2, 0, 1))

        # get_img = lambda index: Image.open(get_path(index)).convert('RGB')
        # get_label = lambda index: self.label[index]
        
        # img = transform(get_img(index))
        # img = torch.unsqueeze(img, 0)
        # label = torch.from_numpy(np.array(get_label(index)))
        # label = torch.unsqueeze(label, 0)
        # return img, label
        return img0, img1, img2, img0_depth, img1_depth, img2_depth, flow01, flow12, flow02, back_flow01, back_flow12, back_flow02
    
    
