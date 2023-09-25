import torch
from torch.utils import data
import numpy as np
import torchvision.transforms as T
import glob

import utils
import random
import pickle

num_classes = 1 + 3

class ReDWeb(data.Dataset):
    def __init__(self, dataset_dir="datasets/ReDWeb_V1", num_images=None):
        self.dataset_dir = dataset_dir
        with open("ReDWeb_list.txt", "r") as f:
            self.img_names = f.readlines()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx][:-1].split(".")[0]
        print(f"{img_name = }")
        img_path = f"{self.dataset_dir}/Imgs/{img_name}.jpg"
        depth_path = f"{self.dataset_dir}/RDs/{img_name}.png"
        img, img_size = utils.get_img(img_path)
        depth, depth_size = utils.get_depth(depth_path)
        if img_size != depth_size:
            depth = T.Resize(img_size)(depth)

        return img, depth


class DIML(data.Dataset):
    def __init__(self, dataset_dir="datasets/DIML", num_images=None, load=False):
        self.dataset_dir = dataset_dir
        with open("DIML_list.txt", "r") as f:
            self.img_names = f.readlines()

    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, idx):
        img_name = self.img_names[idx][:-1].split(".")[0]
        img0_path = f"{self.dataset_dir}/train/LR/outleft/{img_name}.png"
        img1_path = f"{self.dataset_dir}/train/LR/outright/{img_name}.png"
        disp0_path = f"{self.dataset_dir}/train/LR/disparity/{img_name}.png"
        print(f"{img0_path = }")

        img0, img_size = utils.get_img(img0_path)
        img1, _ = utils.get_img(img1_path)
        disp0, disp_size = utils.get_disparity(disp0_path)
        if img_size != disp_size:
            disp0 = T.Resize(img_size)(disp0)

        return img0, img1, disp0


class AugmentedDataset(data.Dataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None, do_flip=True):
        self.normalize_dataset = normalize_dataset
        self.crop_size = crop_size
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        if size is not None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize(size),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def getitem_from_npz(self, npz_filename, group_npz_filename, random_group, idx):
        # print(f"{npz_filename = }, {group_npz_filename = }")
        try:
            npz_file = np.load(npz_filename)
            augment_img = npz_file["augment_img"]
            augment_flow_type = npz_file["augment_flow_type"]
            _, h, w = npz_file["img_depth_flow"].shape
            img_depth_flow = npz_file["img_depth_flow"]

            group_npz_file = np.load(group_npz_filename)
            group_img_depth_flow = group_npz_file["img_depth_flow"]
        except:
            return self.__getitem__((idx + 1) % self.__len__())

        if random_group == 0:
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[4:7]
        elif random_group == 1:
            img0 = group_img_depth_flow[4:7]
            img0_depth = group_img_depth_flow[7:8]
            img1 = group_img_depth_flow[8:11]
        elif random_group == 2:
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[8:11]
        img0 = self.transform(img0.transpose(1, 2, 0))
        img1 = self.transform(img1.transpose(1, 2, 0))
        img0_depth = self.transform(img0_depth.transpose(1, 2, 0))

        if self.normalize_dataset:
            if augment_img == 0:
                img_depth_flow[4] = img_depth_flow[4] / h
                img_depth_flow[5] = img_depth_flow[5] / w
                img_depth_flow[3] = img_depth_flow[3] / 100
            else:
                img_depth_flow[0] = img_depth_flow[0] / h
                img_depth_flow[1] = img_depth_flow[1] / w
                img_depth_flow[7] = img_depth_flow[7] / 100

        img_depth_flow = self.transform(img_depth_flow.transpose(1, 2, 0))

        if augment_img == 0:
            img0 = img_depth_flow[0:3]
            img0_depth = img_depth_flow[3:4]
            flow = img_depth_flow[4:6]
        else:
            flow = img_depth_flow[0:2]
            img1 = img_depth_flow[4:7]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:
                img0 = torch.flip(img0, (2,))
                img1 = torch.flip(img1, (2,))
                img0_depth = torch.flip(img0_depth, (2,))
                flow = torch.flip(flow, (2,))
                flow[0] = flow[0] * -1.0

            if np.random.rand() < self.v_flip_prob:
                img0 = torch.flip(img0, (1,))
                img1 = torch.flip(img1, (1,))
                img0_depth = torch.flip(img0_depth, (1,))
                flow = torch.flip(flow, (1,))
                flow[1] = flow[1] * -1.0

        if self.crop_size is not None:
            y0 = np.random.randint(0, h - self.crop_size[0] + 1)
            x0 = np.random.randint(0, w - self.crop_size[1] + 1)

            img0 = img0[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img1 = img1[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img0_depth = img0_depth[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]


        label_type = max(0, augment_flow_type - 4)
        label = torch.zeros(num_classes)
        label[label_type] = 1
        return img0, img1, flow, img0_depth, label


class DepthToFlowDataset(data.Dataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None, do_flip=True):
        self.normalize_dataset = normalize_dataset
        self.crop_size = crop_size
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        if size is not None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize(size),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def getitem_from_npz(self, group_npz_filename, random_group, idx):
        # print(f"{npz_filename = }, {group_npz_filename = }")
        try:
            group_npz_file = np.load(group_npz_filename)
            group_img_depth_flow = group_npz_file["img_depth_flow"]
        except:
            return self.__getitem__((idx + 1) % self.__len__())
        if random_group == 0:
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[4:7]
            flow = group_img_depth_flow[12:14]
        elif random_group == 1:
            img0 = group_img_depth_flow[4:7]
            img0_depth = group_img_depth_flow[7:8]
            img1 = group_img_depth_flow[8:11]
            flow = group_img_depth_flow[16:18]
        elif random_group == 2:
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[8:11]
            flow = group_img_depth_flow[20:22]
        img0 = self.transform(img0.transpose(1, 2, 0))
        img1 = self.transform(img1.transpose(1, 2, 0))
        img0_depth = self.transform(img0_depth.transpose(1, 2, 0))
        flow = self.transform(flow.transpose(1, 2, 0))

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:
                img0 = torch.flip(img0, (2,))
                img1 = torch.flip(img1, (2,))
                img0_depth = torch.flip(img0_depth, (2,))
                flow = torch.flip(flow, (2,))
                flow[0] = flow[0] * -1.0

            if np.random.rand() < self.v_flip_prob:
                img0 = torch.flip(img0, (1,))
                img1 = torch.flip(img1, (1,))
                img0_depth = torch.flip(img0_depth, (1,))
                flow = torch.flip(flow, (1,))
                flow[1] = flow[1] * -1.0

        if self.crop_size is not None:
            y0 = np.random.randint(0, h - self.crop_size[0] + 1)
            x0 = np.random.randint(0, w - self.crop_size[1] + 1)

            img0 = img0[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img1 = img1[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img0_depth = img0_depth[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        label_type = 0
        label = torch.zeros(num_classes)
        label[label_type] = 1
        return img0, img1, flow, img0_depth, label


class AugmentedDIML(AugmentedDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        # return 1505
        return 1505 * 2

    def __getitem__(self, idx):
        dataset_dir = f"datasets/AugmentedDatasets/DIML"
        random_group = np.random.randint(0, 3)
        random_augment = np.random.randint(0, 12)
        random_set = np.random.randint(1, 3)
        npz_filename = f"{dataset_dir}/{idx}/{random_group}_{random_augment}_{random_set}.npz"
        # group_npz_filename = f"{dataset_dir}/{idx + 1505}/group.npz"
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(npz_filename, group_npz_filename, random_group, idx)


class AugmentedReDWeb(AugmentedDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        return 1698 * 2

    def __getitem__(self, idx):
        dataset_dir = f"datasets/AugmentedDatasets/ReDWeb"
        random_group = np.random.randint(0, 3)
        random_augment = np.random.randint(0, 12)
        random_set = np.random.randint(1, 3)
        npz_filename = f"{dataset_dir}/{idx}/{random_group}_{random_augment}_{random_set}.npz"
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(npz_filename, group_npz_filename, random_group, idx)

