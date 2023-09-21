import torch
from torch.utils import data
import numpy as np
import torchvision.transforms as T
import glob

import utils
import random
import pickle

from my_cuda_loffsi.fw import FW
from preprocess_continuous import augment_flow

num_classes = 1 + 3

class ReDWeb(data.Dataset):
    def __init__(self, dataset_dir, num_images=None):
        self.dataset_dir = dataset_dir
        self.img_paths = glob.glob(f"{dataset_dir}/Imgs/*")
        random.shuffle(self.img_paths)
        # with open("ReDWeb_img_paths.pkl", "rb") as fp:
        #     self.img_paths = pickle.load(fp)
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

class FiltedReDWeb(data.Dataset):
    def __init__(self, dataset_dir, num_images=None):
        self.dataset_dir = dataset_dir
        with open("filted_ReDWeb_list.txt", "r") as f:
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
    def __init__(self, dataset_dir, num_images=None, load=False):
        self.dataset_dir = dataset_dir
        with open("DIML_list.txt", "r") as f:
            self.img_names = f.readlines()

    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, idx):
        # print(f"{idx = }")
        img_name = self.img_names[idx][:-1].split(".")[0]
        print(f"{img_name = }")
        img0_path = f"{self.dataset_dir}/train/LR/outleft/{img_name}.png"
        img1_path = f"{self.dataset_dir}/train/LR/outright/{img_name}.png"
        disp0_path = f"{self.dataset_dir}/train/LR/disparity/{img_name}.png"

        img0, img_size = utils.get_img(img0_path)
        img1, _ = utils.get_img(img1_path)
        disp0, disp_size = utils.get_disparity(disp0_path)
        disp1 = None
        if img_size != disp_size:
            disp0 = T.Resize(img_size)(disp0)

        return img0, img1, disp0, disp1


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
            if npz_filename is not None:
                npz_file = np.load(npz_filename)
                # augment_img = npz_file["augment_img"]
                augment_flow_type = npz_file["augment_flow_type"]
                _, h, w = npz_file["img_depth_flow"].shape
                img_depth_flow = npz_file["img_depth_flow"]

            group_npz_file = np.load(group_npz_filename)
            group_img_depth_flow = group_npz_file["img_depth_flow"]
        except:
            return self.__getitem__((idx + 1) % self.__len__())

        if random_group == 0:
            # 0 -> 1
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[4:7]
            img1_depth = group_img_depth_flow[7:8]
            flow = group_img_depth_flow[24:26]
            back_flow = group_img_depth_flow[26:28]
        elif random_group == 1:
            # 1 -> 2
            img0 = group_img_depth_flow[4:7]
            img0_depth = group_img_depth_flow[7:8]
            img1 = group_img_depth_flow[8:11]
            img1_depth = group_img_depth_flow[11:12]
            flow = group_img_depth_flow[28:30]
            back_flow = group_img_depth_flow[30:32]
        elif random_group == 2:
            # 0 -> 2'
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[16:19]
            img1_depth = group_img_depth_flow[19:20]
            flow = group_img_depth_flow[32:34]
            back_flow = group_img_depth_flow[34:36]
        elif random_group == 3:
            # 0 -> 3
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[12:15]
            img1_depth = group_img_depth_flow[15:16]
            flow = group_img_depth_flow[36:38]
            back_flow = group_img_depth_flow[38:40]
        elif random_group == 4:
            # 1 -> 3'
            img0 = group_img_depth_flow[4:7]
            img0_depth = group_img_depth_flow[7:8]
            img1 = group_img_depth_flow[20:23]
            img1_depth = group_img_depth_flow[23:24]
            flow = group_img_depth_flow[40:42]
            back_flow = group_img_depth_flow[42:44]

        img0 = self.transform(img0.transpose(1, 2, 0))
        img1 = self.transform(img1.transpose(1, 2, 0))
        img0_depth = self.transform(img0_depth.transpose(1, 2, 0))
        img1_depth = self.transform(img1_depth.transpose(1, 2, 0))
        flow = self.transform(flow.transpose(1, 2, 0))
        back_flow = self.transform(back_flow.transpose(1, 2, 0))

        if npz_filename is not None:
            assert npz_filename[-5] == '1' or npz_filename[-5] == '2', "wrong npz_filename"
            if self.normalize_dataset:
                if npz_filename[-5] == '1':
                    img_depth_flow[4] = img_depth_flow[4] / h
                    img_depth_flow[5] = img_depth_flow[5] / w
                    img_depth_flow[6] = img_depth_flow[6] / h
                    img_depth_flow[7] = img_depth_flow[7] / w
                    img_depth_flow[3] = img_depth_flow[3] / 100
                elif npz_filename[-5] == '2':
                    img_depth_flow[0] = img_depth_flow[0] / h
                    img_depth_flow[1] = img_depth_flow[1] / w
                    img_depth_flow[2] = img_depth_flow[2] / h
                    img_depth_flow[3] = img_depth_flow[3] / w
                    img_depth_flow[7] = img_depth_flow[7] / 100

            img_depth_flow = self.transform(img_depth_flow.transpose(1, 2, 0))

            if npz_filename[-5] == '1':
                img0 = img_depth_flow[0:3]
                img0_depth = img_depth_flow[3:4]
                flow = img_depth_flow[4:6]
                back_flow = img_depth_flow[6:8]
            elif npz_filename[-5] == '2':
                flow = img_depth_flow[0:2]
                back_flow = img_depth_flow[2:4]
                img1 = img_depth_flow[4:7]
                img1_depth = img_depth_flow[7:8]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:
                img0 = torch.flip(img0, (2,))
                img1 = torch.flip(img1, (2,))
                img0_depth = torch.flip(img0_depth, (2,))
                img1_depth = torch.flip(img1_depth, (2,))
                flow = torch.flip(flow, (2,))
                flow[0] = flow[0] * -1.0
                back_flow = torch.flip(back_flow, (2,))
                back_flow[0] = back_flow[0] * -1.0

            if np.random.rand() < self.v_flip_prob:
                img0 = torch.flip(img0, (1,))
                img1 = torch.flip(img1, (1,))
                img0_depth = torch.flip(img0_depth, (1,))
                img1_depth = torch.flip(img1_depth, (1,))
                flow = torch.flip(flow, (1,))
                flow[1] = flow[1] * -1.0
                back_flow = torch.flip(back_flow, (1,))
                back_flow[1] = back_flow[1] * -1.0

        if self.crop_size is not None:
            y0 = np.random.randint(0, h - self.crop_size[0] + 1)
            x0 = np.random.randint(0, w - self.crop_size[1] + 1)

            img0 = img0[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img1 = img1[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img0_depth = img0_depth[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img1_depth = img1_depth[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            back_flow = back_flow[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        label_type = 0 if npz_filename is None else max(0, augment_flow_type - 4)
        label = torch.zeros(num_classes)
        label[label_type] = 1
        return img0, img1, flow, back_flow, img0_depth, img1_depth, label
        # return img0, img1, flow, img0_depth, label


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
            # 0 -> 1
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[4:7]
            img1_depth = group_img_depth_flow[7:8]
            flow = group_img_depth_flow[24:26]
            back_flow = group_img_depth_flow[26:28]
        elif random_group == 1:
            # 1 -> 2
            img0 = group_img_depth_flow[4:7]
            img0_depth = group_img_depth_flow[7:8]
            img1 = group_img_depth_flow[8:11]
            img1_depth = group_img_depth_flow[11:12]
            flow = group_img_depth_flow[28:30]
            back_flow = group_img_depth_flow[30:32]
        elif random_group == 2:
            # 0 -> 2'
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[16:19]
            img1_depth = group_img_depth_flow[19:20]
            flow = group_img_depth_flow[32:34]
            back_flow = group_img_depth_flow[34:36]
        elif random_group == 3:
            # 0 -> 3
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[12:15]
            img1_depth = group_img_depth_flow[15:16]
            flow = group_img_depth_flow[36:38]
            back_flow = group_img_depth_flow[38:40]
        elif random_group == 4:
            # 1 -> 3'
            img0 = group_img_depth_flow[4:7]
            img0_depth = group_img_depth_flow[7:8]
            img1 = group_img_depth_flow[20:23]
            img1_depth = group_img_depth_flow[23:24]
            flow = group_img_depth_flow[40:42]
            back_flow = group_img_depth_flow[42:44]

        img0 = self.transform(img0.transpose(1, 2, 0))
        img1 = self.transform(img1.transpose(1, 2, 0))
        img0_depth = self.transform(img0_depth.transpose(1, 2, 0))
        img1_depth = self.transform(img1_depth.transpose(1, 2, 0))
        flow = self.transform(flow.transpose(1, 2, 0))
        back_flow = self.transform(back_flow.transpose(1, 2, 0))
        # print(f"{img0.shape = }")
        # print(f"{img0_depth.shape = }")
        # print(f"{flow.shape = }")

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:
                img0 = torch.flip(img0, (2,))
                img1 = torch.flip(img1, (2,))
                img0_depth = torch.flip(img0_depth, (2,))
                img1_depth = torch.flip(img1_depth, (2,))
                flow = torch.flip(flow, (2,))
                flow[0] = flow[0] * -1.0
                back_flow = torch.flip(back_flow, (2,))
                back_flow[0] = back_flow[0] * -1.0

            if np.random.rand() < self.v_flip_prob:
                img0 = torch.flip(img0, (1,))
                img1 = torch.flip(img1, (1,))
                img0_depth = torch.flip(img0_depth, (1,))
                img1_depth = torch.flip(img1_depth, (1,))
                flow = torch.flip(flow, (1,))
                flow[1] = flow[1] * -1.0
                back_flow = torch.flip(back_flow, (1,))
                back_flow[1] = back_flow[1] * -1.0

        if self.crop_size is not None:
            y0 = np.random.randint(0, h - self.crop_size[0] + 1)
            x0 = np.random.randint(0, w - self.crop_size[1] + 1)

            img0 = img0[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img1 = img1[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img0_depth = img0_depth[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img1_depth = img1_depth[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            back_flow = back_flow[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        label_type = 0
        label = torch.zeros(num_classes)
        label[label_type] = 1
        return img0, img1, flow, back_flow, img0_depth, img1_depth, label
        # return img0, img1, flow, img0_depth, label


class AugmentingDataset(data.Dataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None, do_flip=True, device=None):
        self.normalize_dataset = normalize_dataset
        self.crop_size = crop_size
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        self.device = device

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
            # 0 -> 1
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[4:7]
            img1_depth = group_img_depth_flow[7:8]
            flow = group_img_depth_flow[24:26]
            back_flow = group_img_depth_flow[26:28]
        elif random_group == 1:
            # 1 -> 2
            img0 = group_img_depth_flow[4:7]
            img0_depth = group_img_depth_flow[7:8]
            img1 = group_img_depth_flow[8:11]
            img1_depth = group_img_depth_flow[11:12]
            flow = group_img_depth_flow[28:30]
            back_flow = group_img_depth_flow[30:32]
        elif random_group == 2:
            # 0 -> 2'
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[16:19]
            img1_depth = group_img_depth_flow[19:20]
            flow = group_img_depth_flow[32:34]
            back_flow = group_img_depth_flow[34:36]
        elif random_group == 3:
            # 0 -> 3
            img0 = group_img_depth_flow[0:3]
            img0_depth = group_img_depth_flow[3:4]
            img1 = group_img_depth_flow[12:15]
            img1_depth = group_img_depth_flow[15:16]
            flow = group_img_depth_flow[36:38]
            back_flow = group_img_depth_flow[38:40]
        elif random_group == 4:
            # 1 -> 3'
            img0 = group_img_depth_flow[4:7]
            img0_depth = group_img_depth_flow[7:8]
            img1 = group_img_depth_flow[20:23]
            img1_depth = group_img_depth_flow[23:24]
            flow = group_img_depth_flow[40:42]
            back_flow = group_img_depth_flow[42:44]

        img0 = self.transform(img0.transpose(1, 2, 0)).to(self.device).float()
        img1 = self.transform(img1.transpose(1, 2, 0)).to(self.device).float()
        img0_depth = self.transform(img0_depth.transpose(1, 2, 0)).to(self.device).float()
        img1_depth = self.transform(img1_depth.transpose(1, 2, 0)).to(self.device).float()
        flow = self.transform(flow.transpose(1, 2, 0)).to(self.device).float()
        back_flow = self.transform(back_flow.transpose(1, 2, 0)).to(self.device).float()

        if np.random.randint(0, 2) == 1:# normal
            augment_flow_type = np.random.randint(0, 3)
        else:
            augment_flow_type = np.random.randint(5, 8)
        set1, set2, _, set3 = augment_flow(img0, img0_depth, img1, img1_depth, flow, back_flow,
                                           device=self.device, augment_flow_type=augment_flow_type)

        if np.random.randint(0, 2) == 1:
            img0 = set1[0]
            img0_depth = set1[1]
            flow = set1[2]
            back_flow = set1[3]
        else:
            flow = set2[2]
            back_flow = set2[3]
            img1 = set2[4]
            img1_depth = set2[5]

        del set1, set2, set3
        img0 = img0.cpu()
        img1 = img1.cpu()
        img0_depth = img0_depth.cpu()
        img1_depth = img1_depth.cpu()
        flow = flow.cpu()
        back_flow = back_flow.cpu()

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:
                img0 = torch.flip(img0, (2,))
                img1 = torch.flip(img1, (2,))
                img0_depth = torch.flip(img0_depth, (2,))
                img1_depth = torch.flip(img1_depth, (2,))
                flow = torch.flip(flow, (2,))
                flow[0] = flow[0] * -1.0
                back_flow = torch.flip(back_flow, (2,))
                back_flow[0] = back_flow[0] * -1.0

            if np.random.rand() < self.v_flip_prob:
                img0 = torch.flip(img0, (1,))
                img1 = torch.flip(img1, (1,))
                img0_depth = torch.flip(img0_depth, (1,))
                img1_depth = torch.flip(img1_depth, (1,))
                flow = torch.flip(flow, (1,))
                flow[1] = flow[1] * -1.0
                back_flow = torch.flip(back_flow, (1,))
                back_flow[1] = back_flow[1] * -1.0

        if self.crop_size is not None:
            y0 = np.random.randint(0, h - self.crop_size[0] + 1)
            x0 = np.random.randint(0, w - self.crop_size[1] + 1)

            img0 = img0[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img1 = img1[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img0_depth = img0_depth[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img1_depth = img1_depth[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            back_flow = back_flow[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        label_type = max(0, augment_flow_type - 4)
        label = torch.zeros(num_classes)
        label[label_type] = 1
        return img0, img1, flow, back_flow, img0_depth, img1_depth, label


class AugmentedDIML(AugmentedDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        # return 1505
        return 1505 * 3

    def __getitem__(self, idx):
        # dataset_dir = f"datasets/AugmentedDatasets/DIML"
        images_dirs = ["dataA", "dataB", "dataC"]
        dataset_dir = f"datasets/AugmentedDatasets/DIML/{images_dirs[int((idx % 1505)/502)]}"
        random_group = np.random.randint(0, 3)
        # random_augment = np.random.randint(0, 12)
        if np.random.randint(0, 2) == 1:
            random_augment = np.random.randint(0, 3) + 1 + np.random.randint(0, 3) * 4
            # (0, 1, 2) + (0, 4, 8) + 1 = 1, 2, 3, 5, 6, 7, 9, 10, 11
        else:
            random_augment = np.random.randint(0, 3) * 4 # 0, 4, 8

        random_set = np.random.randint(1, 3)
        npz_filename = f"{dataset_dir}/{idx}/{random_group}_{random_augment}_{random_set}.npz"
        # group_npz_filename = f"{dataset_dir}/{idx + 1505}/group.npz"
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(npz_filename, group_npz_filename, random_group, idx)


class FlowDIML(DepthToFlowDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        # return 1505
        return 1505 * 2

    def __getitem__(self, idx):
        dataset_dir = f"datasets/AugmentedDatasets/DIML"
        random_group = np.random.randint(0, 3)
        # group_npz_filename = f"{dataset_dir}/{idx + 1505}/group.npz"
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(group_npz_filename, random_group, idx)


class VEMDIML(DepthToFlowDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        # return 1505
        return 1505 * 2

    def __getitem__(self, idx):
        dataset_dir = f"datasets/AugmentedDatasets/DIML"
        random_group = 1
        # group_npz_filename = f"{dataset_dir}/{idx + 1505}/group.npz"
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(group_npz_filename, random_group, idx)


class FlowFiltedReDWeb(DepthToFlowDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        return 1698 * 2

    def __getitem__(self, idx):
        dataset_dir = f"datasets/AugmentedDatasets/filted_ReDWeb"
        random_group = np.random.randint(0, 3)
        # group_npz_filename = f"{dataset_dir}/{idx + 3600}/group.npz"
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(group_npz_filename, random_group, idx)

class VEMFiltedReDWeb(DepthToFlowDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        return 1698 * 2

    def __getitem__(self, idx):
        dataset_dir = f"datasets/AugmentedDatasets/filted_ReDWeb"
        random_group = 1
        # group_npz_filename = f"{dataset_dir}/{idx + 3600}/group.npz"
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(group_npz_filename, random_group, idx)


class AugmentedFiltedReDWeb(AugmentedDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        return 1698 * 2

    def __getitem__(self, idx):
        dataset_dir = f"datasets/AugmentedDatasets/filted_ReDWeb"
        random_group = np.random.randint(0, 3)
        # random_augment = np.random.randint(0, 12)
        if np.random.randint(0, 2) == 1:
            random_augment = np.random.randint(0, 3) + 1 + np.random.randint(0, 3) * 4
            # (0, 1, 2) + (0, 4, 8) + 1 = 1, 2, 3, 5, 6, 7, 9, 10, 11
        else:
            random_augment = np.random.randint(0, 3) * 4 # 0, 4, 8
        random_set = np.random.randint(1, 3)
        npz_filename = f"{dataset_dir}/{idx}/{random_group}_{random_augment}_{random_set}.npz"
        # group_npz_filename = f"{dataset_dir}/{idx + 1698}/group.npz"
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(npz_filename, group_npz_filename, random_group, idx)


class AugmentedVEMDIML(DepthToFlowDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        # return 1505
        return 1505 * 2

    def __getitem__(self, idx):
        dataset_dir = f"datasets/AugmentedDatasets/DIML"
        random_group = 1
        random_augment = np.random.randint(0, 12)
        random_set = np.random.randint(1, 3)
        npz_filename = f"{dataset_dir}/{idx}/{random_group}_{random_augment}_{random_set}.npz"
        # group_npz_filename = f"{dataset_dir}/{idx + 1505}/group.npz"
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(npz_filename, group_npz_filename, random_group, idx)



class dCOCO(DepthToFlowDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        return 20000 * 5

    def __getitem__(self, idx):
        dataset_dir_idx = int(idx / int((20000 + 3) // 4)) + 1
        dataset_dir = f"datasets/AugmentedDatasets/dCOCO/data{dataset_dir_idx}"
        random_group = 3 # 0 -> 3
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(group_npz_filename, random_group, idx)

class dCOCO2(DepthToFlowDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        return 20000 * 5

    def __getitem__(self, idx):
        dataset_dir_idx = int(idx / int((20000 + 3) // 4)) + 1
        dataset_dir = f"datasets/AugmentedDatasets/dCOCO/data{dataset_dir_idx}"
        random_group = np.random.randint(0, 2) * 2 + 1 # 1: 1 -> 2, 3: 0 -> 3
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(group_npz_filename, random_group, idx)


class ExtendeddCOCO(DepthToFlowDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        return 20000 * 5

    def __getitem__(self, idx):
        dataset_dir_idx = int(idx / int((20000 + 3) // 4)) + 1
        dataset_dir = f"datasets/AugmentedDatasets/dCOCO/data{dataset_dir_idx}"
        random_group = np.random.randint(0, 5)
        group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
        return self.getitem_from_npz(group_npz_filename, random_group, idx)


class AugmenteddCOCO(AugmentedDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size)

    def __len__(self):
        return 20000 * 5

    def __getitem__(self, idx):
        try:
            dataset_dir_idx = int(idx / int((20000 + 3) // 4)) + 1
            dataset_dir = f"datasets/AugmentedDatasets/dCOCO/data{dataset_dir_idx}"
            random_group = np.random.randint(0, 5)
            random_set = np.random.randint(1, 3)
            npz_filename = f"{dataset_dir}/{idx}/{random_group}_{random_set}.npz"
            group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
            if np.random.randint(0, 2) == 1:
                res = self.getitem_from_npz(None, group_npz_filename, random_group, idx)
            res = self.getitem_from_npz(npz_filename, group_npz_filename, random_group, idx)
        except:
            res = self.__getitem__((idx + 1) % self.__len__())
        return res

class AugmentingdCOCO(AugmentingDataset):
    def __init__(self, normalize_dataset=True, size=None, crop_size=None, device=None):
        super().__init__(normalize_dataset=normalize_dataset, size=size, crop_size=crop_size, device=device)

    def __len__(self):
        return 20000 * 5

    def __getitem__(self, idx):
        try:
            dataset_dir_idx = int(idx / int((20000 + 3) // 4)) + 1
            dataset_dir = f"datasets/AugmentedDatasets/dCOCO/data{dataset_dir_idx}"
            random_group = np.random.randint(0, 5)
            group_npz_filename = f"{dataset_dir}/{idx}/group.npz"
            res = self.getitem_from_npz(group_npz_filename, random_group, idx)
        except:
            res = self.__getitem__((idx + 1) % self.__len__())
        return res
