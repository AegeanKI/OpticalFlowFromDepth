from classifier import Classifier, RAFTClassifier
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda import is_available
import torchvision.transforms as T
from torch import device, from_numpy
import torch
# from dataloader import dCOCODataset, KITTIDataset
from dataloader import AugmentedReDWeb, AugmentedMiddlebury
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import utils
import time
import json
import sys

import gc
# from augment import augment_flow
# import nvidia_smi
import cv2
from my_cuda.fw import FW
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from argparse import ArgumentParser, BooleanOptionalAction

# def get_loaders(batch_size):
#     train_dataset = dCOCOLoader("train")
#     # test_dataset = dCOCOLoader("test")
#     train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
#     # test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
#     return train_loader, None
#     # return train_loader, test_loader
#     # return train_dataset, test_dataset

# def get_loaders():

#     train_loader = DataLoader(train_dataset, 1, shuffle=True) # batch_size already 32
#     return train_loader

# def get_cuda_device():
#     return device("cuda:2" if is_available() else "cpu")

# def get_data(img_depth_flow):
#     img_depth_flow = img_depth_flow.squeeze()

#     img0 = img_depth_flow[:, 0:3]
#     img1 = img_depth_flow[:, 3:6]
#     img2 = img_depth_flow[:, 6:9]
#     img0_depth = img_depth_flow[:, 9:10]
#     img1_depth = img_depth_flow[:, 10:11]
#     img2_depth = img_depth_flow[:, 11:12]
#     concat_img = img_depth_flow[:, 12:15]
#     flow01 = img_depth_flow[:, 15:17]
#     flow12 = img_depth_flow[:, 17:19]
#     flow02 = img_depth_flow[:, 19:21]
#     back_flow01 = img_depth_flow[:, 21:23]
#     back_flow12 = img_depth_flow[:, 23:25]
#     back_flow02 = img_depth_flow[:, 25:27]

#     return (img0, img1, img2, img0_depth, img1_depth, img2_depth,
#             flow01, flow12, flow02, back_flow01, back_flow12, back_flow02)

def my_collate(batch):
    img_depth_flow = [torch.tensor(item[0]) for item in batch]
    augment_img = [item[1] for item in batch]
    augment_flow_type = [item[2] for item in batch]
    return [img_depth_flow, augment_img, augment_flow_type]

def read_args():
    parser = ArgumentParser()

    # parser.add_argument('--mono', default=False, type=bool, action=BooleanOptionalAction)
    parser.add_argument('--use_small', default=False, type=bool, action=BooleanOptionalAction)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--output_dim', type=int)
    parser.add_argument('--use_depth_in_classifier', default=False, action=BooleanOptionalAction)
    parser.add_argument('--use_dropout_in_encoder', default=False, type=bool, action=BooleanOptionalAction)
    parser.add_argument('--use_dropout_in_classify', default=False, type=bool, action=BooleanOptionalAction)
    # parser.add_argument('--use_pooling', default=True, type=bool, action=BooleanOptionalAction)
    parser.add_argument('--use_average_pooling', default=False, type=bool, action=BooleanOptionalAction)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--dataset')
    parser.add_argument('--image_size')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--min_lr', default=0.00002, type=float)
    parser.add_argument('--lr_decay', default=0.00002, type=float)
    parser.add_argument('--max_epoch', default=10000, type=int)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--normalize_dataset', default=True, type=bool, action=BooleanOptionalAction)
        
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    torch.set_printoptions(precision=4)

    args = read_args()
    print(args)
    
    cur_time = time.time()
    os.makedirs(f"outputs/models/{cur_time}")
    with open(f"outputs/models/{cur_time}/args.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    lr = args.lr
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"{device = }")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if not args.use_depth_in_classifier:
        model = Classifier(device=device, output_dim=args.output_dim,
                           dropout=args.dropout, use_small=args.use_small,
                           use_dropout_in_encoder=args.use_dropout_in_encoder,
                           use_dropout_in_classify=args.use_dropout_in_classify,
                           # use_pooling=args.use_pooling,
                           use_average_pooling=args.use_average_pooling)
    else:
        model = RAFTClassifier(device=device, output_dim=args.output_dim,
                               dropout=args.dropout, use_small=args.use_small,
                               use_dropout_in_encoder=args.use_dropout_in_encoder,
                               use_dropout_in_classify=args.use_dropout_in_classify,
                               # use_pooling=args.use_pooling,
                               use_average_pooling=args.use_average_pooling)
    model.to(device)
    # model.load_state_dict(torch.load("output/models/first.pt"))
    loss_func = CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=lr)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0001, eps=1e-8)

    test_split = 0.2
    random_seed = 12345
    # random_seed = 23456
    save_to_test = False

    if args.dataset == "AugmentedReDWeb":
        dataset = AugmentedReDWeb(normalize_dataset=args.normalize_dataset, size=args.image_size)
    elif args.dataset == "AugmentedMiddlebury":
        dataset = AugmentedMiddlebury(normalize_dataset=args.normalize_dataset, size=args.image_size)
    dataset_size = len(dataset)
    
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    
    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, test_indices = indices[split:dataset_size], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              # collate_fn=my_collate,
                              num_workers=0,
                              sampler=train_sampler)
    test_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                              # collate_fn=my_collate,
                             num_workers=0,
                             sampler=test_sampler)


    best_train_acc = 0
    for epoch in range(args.max_epoch):
        total = 0
        correct = 0
        model.train()
        for batch_idx, (_, _, flow, depth, label) in enumerate(tqdm(train_loader)):
            flow = flow.to(device)
            depth = depth.to(device)
            label = label.to(device)
            flow_depth = torch.cat((flow, depth), axis=1).float().to(device)

            optimizer.zero_grad()
            if not args.use_depth_in_classifier:
                predict1 = model(flow)
            else:
                predict1 = model(flow_depth)
            loss1 = loss_func(predict1, label)
            loss1.backward()
            optimizer.step()

            predict1 = torch.nn.Softmax(dim=1)(predict1)
            total = total + args.batch_size
            correct = correct + (torch.max(predict1, dim=1).indices ==
                                 torch.max(label, dim=1).indices).sum().item()
            # break

        train_acc = correct / total
        print(f"    train accuracy = {correct} / {total} = {correct / total}")

        total = 0
        correct = 0
        confusion_matrix = np.zeros((args.num_classes, args.num_classes))
        model.eval()
        for batch_idx, (_, _, flow, depth, label) in enumerate(tqdm(test_loader)):
            flow = flow.to(device)
            depth = depth.to(device)
            label = label.to(device)
            flow_depth = torch.cat((flow, depth), axis=1).float().to(device)

            optimizer.zero_grad()
            if not args.use_depth_in_classifier:
                predict1 = model(flow)
            else:
                predict1 = model(flow_depth)

            predict1 = torch.nn.Softmax(dim=1)(predict1)
            total = total + args.batch_size
            correct = correct + (torch.max(predict1, dim=1).indices ==
                                 torch.max(label, dim=1).indices).sum().item()

            for p, l in zip(torch.max(predict1, dim=1).indices, torch.max(label, dim=1).indices):
                confusion_matrix[p, l] = confusion_matrix[p, l] + 1
            # break
        test_acc = correct / total
        print(f"                    test accuracy = {correct} / {total} = {correct / total}")
        if train_acc >= best_train_acc:
            best_train_acc = train_acc
            model_name = type(model).__name__
            print(f"{confusion_matrix = }")
            torch.save(model.state_dict(), f"outputs/models/{cur_time}/{train_acc=:.3f}_{test_acc=:.3f}.pt")
        # break


