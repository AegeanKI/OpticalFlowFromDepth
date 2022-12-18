import torch
from torch.utils import data
import numpy as np
from torchvision import transforms

num_classes = 1 + 3

class AugmentedReDWeb(data.Dataset):
    def __init__(self, normalize_dataset=True, size=None):
        # self.mode = 
        # self.device = device
        self.normalize_dataset = normalize_dataset

        if size is not None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return 3600
        # return 1200


    def __getitem__(self, idx):
        images_dirs = ["data4", "data6", "data7"]
        dataset_dir = f"datasets/AugmentedReDWeb/{images_dirs[int(idx/1200)]}"
        random_group = np.random.randint(0, 3)
        random_augment = np.random.randint(0, 12)
        random_set = np.random.randint(1, 3)
        npz_filename = f"{idx}/{random_group}_{random_augment}_{random_set}.npz"
        npz_file = np.load(f"{dataset_dir}/{npz_filename}")
        augment_img = npz_file["augment_img"]
        augment_flow_type = npz_file["augment_flow_type"]
        _, h, w = npz_file["img_depth_flow"].shape
        img_depth_flow = npz_file["img_depth_flow"]

        group_npz_filename = f"{idx}/group.npz"
        group_npz_file = np.load(f"{dataset_dir}/{group_npz_filename}")
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
        label = torch.zeros(num_classes)
        label[label_type] = 1
        return img0, img1, flow, img0_depth, label


