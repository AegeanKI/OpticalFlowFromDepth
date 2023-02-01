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
    def __init__(self, dataset_dir, num_images=None):
        self.dataset_dir = dataset_dir
        self.img_paths = glob.glob(f"{dataset_dir}/Imgs/*")
        random.shuffle(self.img_paths)
        with open("ReDWeb_img_paths.pkl", "rb") as fp:
            self.img_paths = pickle.load(fp)
        if num_images and len(self.img_paths) > num_images:
            self.img_paths = self.img_paths[:num_images]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        depth_path = img_path.replace("jpg", "png").replace("Imgs", "RDs")
        img, img_size = utils.get_img(img_path)
        depth, depth_size = utils.get_depth(depth_path)
        if img_size != depth_size:
            depth = T.Resize(img_size)(depth)

        return img, depth

# class Middlebury(data.Dataset):
#     def __init__(self, dataset_dir, num_images=None):
#         self.dataset_dir = dataset_dir
#         self.img_paths = glob.glob(f"{dataset_dir}/*")
#         random.shuffle(self.img_paths)
#         if num_images and len(self.img_paths) > num_images:
#             self.img_paths = self.img_paths[:num_images]

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, idx):
#         img0_path = f"{self.img_paths[idx]}/im0.png"
#         img1_path = f"{self.img_paths[idx]}/im1.png"
#         disp0_path = f"{self.img_paths[idx]}/disp0.pfm"
#         disp1_path = f"{self.img_paths[idx]}/disp1.pfm"

#         img0, img_size = utils.get_img(img0_path)
#         img1, _ = utils.get_img(img1_path)
#         disp0, disp_size = utils.get_disparity(disp0_path)
#         disp1, _ = utils.get_disparity(disp1_path)

#         if img_size != disp_size:
#             disp0 = T.Resize(img_size)(disp0)
#             disp1 = T.Resize(img_size)(disp1)

#         return img0, img1, disp0, disp1


class DIML(data.Dataset):
    def __init__(self, dataset_dir, num_images=None, load=False):
        self.dataset_dir = dataset_dir
        self.img_paths = glob.glob(f"{dataset_dir}/train/LR/outleft/*")
        random.shuffle(self.img_paths)
        # with open("DIML_img_paths.pkl", "rb") as fp:
        #     self.img_paths = pickle.load(fp)
        if num_images and len(self.img_paths) > num_images:
            self.img_paths = self.img_paths[:num_images]

    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        # print(f"{idx = }")
        img0_path = self.img_paths[idx]
        img1_path = img0_path.replace("outleft", "outright")
        disp0_path = img0_path.replace("outleft", "disparity")

        img0, img_size = utils.get_img(img0_path)
        img1, _ = utils.get_img(img1_path)
        disp0, disp_size = utils.get_disparity(disp0_path)
        disp1 = None
        if img_size != disp_size:
            disp0 = T.Resize(img_size)(disp0)

        return img0, img1, disp0, disp1


class ETH3D(data.Dataset):
    def __init__(self, dataset_dir, num_images=None):
        self.dataset_dir = dataset_dir
        self.img_paths = glob.glob(f"{dataset_dir}/*/stereo_pairs/*")
        # print(f"{self.img_path = }")
        random.shuffle(self.img_paths)
        if num_images and len(self.img_paths) > num_images:
            self.img_paths = self.img_paths[:num_images]

    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        img0_path = f"{self.img_paths[idx]}/im0.png"
        img1_path = f"{self.img_paths[idx]}/im1.png"
        disp0_path = f"{self.img_paths[idx]}/disp0GT.pfm"
        disp1_path = f"{self.img_paths[idx]}/disp1GT.pfm"

        img0, img_size = utils.get_img(img0_path)
        img1, _ = utils.get_img(img1_path)
        disp0, disp_size = utils.get_disparity(disp0_path)
        disp1, _ = utils.get_disparity(disp1_path)
        if img_size != disp_size:
            disp0 = T.Resize(img_size)(disp0)
            disp1 = T.Resize(img_size)(disp1)

        return img0, img1, disp0, disp1


class AugmentedDataset(data.Dataset):
    def __init__(self, normalize_dataset=True, size=None):
        self.normalize_dataset = normalize_dataset

        if size is not None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize(size),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
            ])

    def getitem_from_npz(self, npz_filename, group_npz_filename, random_group):
        # print(f"{npz_filename = }, {group_npz_filename = }")
        npz_file = np.load(npz_filename)
        augment_img = npz_file["augment_img"]
        augment_flow_type = npz_file["augment_flow_type"]
        _, h, w = npz_file["img_depth_flow"].shape
        img_depth_flow = npz_file["img_depth_flow"]

        group_npz_file = np.load(group_npz_filename)
        group_img_depth_flow = group_npz_file["img_depth_flow"]
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

        label_type = max(0, augment_flow_type - 4)
        # label = utils.one_hot(label_type, num_classes)
        label = torch.zeros(num_classes)
        label[label_type] = 1
        return img0, img1, flow, img0_depth, label



class AugmentedReDWeb(AugmentedDataset):
    def __init__(self, normalize_dataset=True, size=None):
        super().__init__(normalize_dataset, size)

    def __len__(self):
        # return 3600
        return 7200

    def __getitem__(self, idx):
        images_dirs = ["dataA", "dataB", "dataC"]
        dataset_dir = f"datasets/AugmentedReDWeb/{images_dirs[int((idx % 3600)/1200)]}"
        random_group = np.random.randint(0, 3)
        random_augment = np.random.randint(0, 12)
        random_set = np.random.randint(1, 3)
        # idx = idx + 3600
        npz_filename = f"{dataset_dir}/{idx}/{random_group}_{random_augment}_{random_set}.npz"
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(npz_filename, group_npz_filename, random_group)


class AugmentedDIML(AugmentedDataset):
    def __init__(self, normalize_dataset=True, size=None):
        super().__init__(normalize_dataset, size)

    def __len__(self):
        return 1505

    def __getitem__(self, idx):
        images_dirs = ["dataA", "dataB", "dataC"]
        dataset_dir = f"datasets/AugmentedDIML/{images_dirs[int(idx/502)]}"
        random_group = np.random.randint(0, 3)
        random_augment = np.random.randint(0, 12)
        random_set = np.random.randint(1, 3)
        npz_filename = f"{dataset_dir}/{idx}/{random_group}_{random_augment}_{random_set}.npz"
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(npz_filename, group_npz_filename, random_group)
