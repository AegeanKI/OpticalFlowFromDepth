from classifier import Classifier, ResnetClassifier, RAFTClassifier
import torch.optim as optim
from torch.optim import Adam, Adagrad, AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda import is_available
import torchvision.transforms as T
from torch import device, from_numpy
import torch
from dataloader import dCOCODataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import utils

import gc
from augment import augment_flow
import nvidia_smi
import cv2
from my_cuda.fw import FW
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# def get_loaders(batch_size):
#     train_dataset = dCOCOLoader("train")
#     # test_dataset = dCOCOLoader("test")
#     train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
#     # test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
#     return train_loader, None
#     # return train_loader, test_loader
#     # return train_dataset, test_dataset

def get_loaders():

    train_loader = DataLoader(train_dataset, 1, shuffle=True) # batch_size already 32
    return train_loader

def get_cuda_device():
    return device("cuda:2" if is_available() else "cpu")

def get_data(img_depth_flow):
    img_depth_flow = img_depth_flow.squeeze()

    img0 = img_depth_flow[:, 0:3]
    img1 = img_depth_flow[:, 3:6]
    img2 = img_depth_flow[:, 6:9]
    img0_depth = img_depth_flow[:, 9:10]
    img1_depth = img_depth_flow[:, 10:11]
    img2_depth = img_depth_flow[:, 11:12]
    concat_img = img_depth_flow[:, 12:15]
    flow01 = img_depth_flow[:, 15:17]
    flow12 = img_depth_flow[:, 17:19]
    flow02 = img_depth_flow[:, 19:21]
    back_flow01 = img_depth_flow[:, 21:23]
    back_flow12 = img_depth_flow[:, 23:25]
    back_flow02 = img_depth_flow[:, 25:27]

    return (img0, img1, img2, img0_depth, img1_depth, img2_depth,
            flow01, flow12, flow02, back_flow01, back_flow12, back_flow02)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    torch.set_printoptions(precision=4)

    nvidia_smi.nvmlInit()
    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(2)
    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle) 
    # print(f"init {info.total = }, {info.free = }, {info.used = }")

    lr = 0.00002
    batch_size = 1
    # size = (640, 480)
    size = (1242, 375) 
    # size = (224, 224)
    # resize = T.Resize(size)
    max_epoch = 10000
    num_classes = 1 + 5

    gpu = 2
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    print(f"{device = }")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    model = Classifier(size=None, device=device, use_small=True)
    model.to(device)
    fw = FW(size, batch_size, device).to(device)
    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle) 
    # print(f"after model {info.total = }, {info.free = }, {info.used = }")

    # model.load_state_dict(torch.load("output/models/first.pt"))
    loss_func = CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=lr)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.00005, eps=1e-8)

    test_split = 0.2
    random_seed = 12345
    # random_seed = int(time.time())
    save_to_test = False
    dataset = dCOCODataset("train", mono=False, device=device)
    dataset_size = len(dataset)
    # dataset_size = len(dataset) // 10 

    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    
    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, test_indices = indices[split:dataset_size], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=0, sampler=test_sampler)

    best_accuracy = 0
    for epoch in range(max_epoch):

        total = 0
        correct = 0
        total_loss = 0
        for batch_idx, img_depth_flow in enumerate(tqdm(train_loader)):
            # img_depth_flow = img_depth_flow.type(torch.float32).to("cuda") 

            (img0, img1, img2, img0_depth, img1_depth, img2_depth,
            flow01, flow12, flow02, back_flow01, back_flow12, back_flow02) = get_data(img_depth_flow)

            batch_size = img0.size(0)
            group = [(img0, img0_depth, img1, img1_depth, flow01, back_flow01),
                     (img1, img1_depth, img2, img2_depth, flow12, back_flow12),
                     (img0, img0_depth, img2, img2_depth, flow02, back_flow02)]

            for group_idx, (imgA, imgA_depth, imgB, imgB_depth, flowAB, back_flowAB) in enumerate(group):
                imgA = imgA.type(torch.float32).to(device)
                imgB = imgB.type(torch.float32).to(device)
                imgA_depth = imgA_depth.type(torch.float32).to(device)
                imgB_depth = imgB_depth.type(torch.float32).to(device)
                flowAB = flowAB.type(torch.float32).to(device)
                back_flowAB = back_flowAB.type(torch.float32).to(device)

                normal_type = utils.get_random(3, 0, False)

                augment_flow_type = utils.get_random(4, 0, False)
                if augment_flow_type >= 1.:
                    # augment_flow_type = augment_flow_type + 2    # 1, 2, 3, 4, 5 -> 3, 4, 5, 6, 7
                    augment_flow_type = augment_flow_type + 4      # 1, 2, 3 -> 5, 6, 7
                else:
                    augment_flow_type = normal_type                # 0 -> 0, 1, 2

                fake_set1, fake_set2, _ = augment_flow(imgA, torch.ones_like(imgA_depth),
                            imgB, torch.ones_like(imgB_depth),
                            flowAB, back_flowAB, device=device, augment_flow_type=augment_flow_type)

                set1, set2, _ = augment_flow(imgA, imgA_depth, imgB, imgB_depth,
                            flowAB, back_flowAB, device=device, augment_flow_type=augment_flow_type)

                label_type = max(0, int(augment_flow_type) - 2) # 0, 1, 2 is normal
                augment_label = torch.zeros((batch_size, num_classes)).to(device)
                augment_label[:, label_type] = 1

                if save_to_test:
                    output_dir = f"testing/train_classifier/train_{label_type}_{epoch}_{batch_idx}_{group_idx}"
                    os.mkdir(output_dir)
                    augment_imgA_img0_depth = set1[1][0, 0].cpu().numpy()
                    augment_imgB_img0_depth = set1[1][0, 0].cpu().numpy()
                    plt.imsave(f"{output_dir}/augment_imgA_img0_depth.png", 1 / augment_imgA_img0_depth, cmap="magma")
                    plt.imsave(f"{output_dir}/augment_imgB_img0_depth.png", 1 / augment_imgB_img0_depth, cmap="magma")

                    fake_augment_imgA_img0 = fake_set1[0][0].permute(1, 2, 0).cpu().numpy()
                    fake_augment_imgB_img0 = fake_set2[0][0].permute(1, 2, 0).cpu().numpy()
                    fake_augment_imgA_img1 = fw(fake_set1[0], fake_set1[2], fake_set1[1])[0].permute(1, 2, 0).cpu().numpy() 
                    fake_augment_imgB_img1 = fw(fake_set2[0], fake_set2[2], fake_set2[1])[0].permute(1, 2, 0).cpu().numpy() 
                    cv2.imwrite(f"{output_dir}/augment_imgA_img0_fake.png", fake_augment_imgA_img0)
                    cv2.imwrite(f"{output_dir}/augment_imgB_img0_fake.png", fake_augment_imgB_img0)
                    cv2.imwrite(f"{output_dir}/augment_imgA_img1_fake.png", fake_augment_imgA_img1)
                    cv2.imwrite(f"{output_dir}/augment_imgB_img1_fake.png", fake_augment_imgB_img1)

                    fake_augment_imgA_flow_color = fake_set1[2][0:1].permute(0, 2, 3, 1).cpu()
                    fake_augment_imgB_flow_color = fake_set2[2][0:1].permute(0, 2, 3, 1).cpu()
                    fake_back_augment_imgA_flow_color = fake_set1[3][0:1].permute(0, 2, 3, 1).cpu()
                    fake_back_augment_imgB_flow_color = fake_set2[3][0:1].permute(0, 2, 3, 1).cpu()
                    _, fake_augment_imgA_flow_color = utils.color_flow(fake_augment_imgA_flow_color)
                    _, fake_augment_imgB_flow_color = utils.color_flow(fake_augment_imgB_flow_color)
                    _, fake_back_augment_imgA_flow_color = utils.color_flow(fake_back_augment_imgA_flow_color)
                    _, fake_back_augment_imgB_flow_color = utils.color_flow(fake_back_augment_imgB_flow_color)
                    cv2.imwrite(f"{output_dir}/augment_imgA_flow_fake.png", fake_augment_imgA_flow_color)
                    cv2.imwrite(f"{output_dir}/augment_imgB_flow_fake.png", fake_augment_imgB_flow_color)
                    cv2.imwrite(f"{output_dir}/back_augment_imgA_flow_fake.png", fake_back_augment_imgA_flow_color)
                    cv2.imwrite(f"{output_dir}/back_augment_imgB_flow_fake.png", fake_back_augment_imgB_flow_color)

                    augment_imgA_img0 = set1[0][0].permute(1, 2, 0).cpu().numpy()
                    augment_imgB_img0 = set2[0][0].permute(1, 2, 0).cpu().numpy()
                    augment_imgA_img1 = fw(set1[0], set1[2], set1[1])[0].permute(1, 2, 0).cpu().numpy() 
                    augment_imgB_img1 = fw(set2[0], set2[2], set2[1])[0].permute(1, 2, 0).cpu().numpy() 
                    cv2.imwrite(f"{output_dir}/augment_imgA_img0.png", augment_imgA_img0)
                    cv2.imwrite(f"{output_dir}/augment_imgB_img0.png", augment_imgB_img0)
                    cv2.imwrite(f"{output_dir}/augment_imgA_img1.png", augment_imgA_img1)
                    cv2.imwrite(f"{output_dir}/augment_imgB_img1.png", augment_imgB_img1)

                    augment_imgA_flow_color = set1[2][0:1].permute(0, 2, 3, 1).cpu()
                    augment_imgB_flow_color = set2[2][0:1].permute(0, 2, 3, 1).cpu()
                    back_augment_imgA_flow_color = set1[3][0:1].permute(0, 2, 3, 1).cpu()
                    back_augment_imgB_flow_color = set2[3][0:1].permute(0, 2, 3, 1).cpu()
                    _, augment_imgA_flow_color = utils.color_flow(augment_imgA_flow_color)
                    _, augment_imgB_flow_color = utils.color_flow(augment_imgB_flow_color)
                    _, back_augment_imgA_flow_color = utils.color_flow(back_augment_imgA_flow_color)
                    _, back_augment_imgB_flow_color = utils.color_flow(back_augment_imgB_flow_color)
                    cv2.imwrite(f"{output_dir}/augment_imgA_flow.png", augment_imgA_flow_color)
                    cv2.imwrite(f"{output_dir}/augment_imgB_flow.png", augment_imgB_flow_color)
                    cv2.imwrite(f"{output_dir}/back_augment_imgA_flow.png", back_augment_imgA_flow_color)
                    cv2.imwrite(f"{output_dir}/back_augment_imgB_flow.png", back_augment_imgB_flow_color)

                # augment_imgA_flow = resize(set1[2])
                # augment_imgB_flow = resize(set2[2])
                augment_imgA_flow = set1[2]
                augment_imgB_flow = set2[2]
                # augment_imgA_depth = set1[1]
                # augment_imgB_depth = set2[1]
                # back_augment_imgA_flow = set1[3]
                # back_augment_imgB_flow = set2[3]

                model.train()
                optimizer.zero_grad()
                predict1 = model(augment_imgA_flow)
                # predict1 = model(torch.cat((augment_imgA_flow, augment_imgA_depth), axis=1))
                loss1 = loss_func(predict1, augment_label)
                loss1.backward()
                optimizer.step()

                predict1 = torch.nn.Softmax(dim=1)(predict1)
                total = total + augment_label.size(0)
                correct = correct + (torch.max(predict1, dim=1).indices ==
                                     torch.max(augment_label, dim=1).indices).sum().item()

                model.train()
                optimizer.zero_grad()
                predict2 = model(augment_imgB_flow)
                # predict2 = model(torch.cat((augment_imgB_flow, augment_imgB_depth), axis=1))
                loss2 = loss_func(predict2, augment_label)
                loss2.backward()
                optimizer.step()

                predict2 = torch.nn.Softmax(dim=1)(predict2)
                total = total + augment_label.size(0)
                correct = correct + (torch.max(predict2, dim=1).indices ==
                                     torch.max(augment_label, dim=1).indices).sum().item()

                # model.train()
                # optimizer.zero_grad()
                # predict3 = model(back_augment_imgA_flow)
                # loss3 = loss_func(predict3, augment_label)
                # loss3.backward()
                # optimizer.step()

                # predict3 = torch.nn.Softmax(dim=1)(predict3)
                # total = total + augment_label.size(0)
                # correct = correct + (torch.max(predict3, dim=1).indices ==
                #                      torch.max(augment_label, dim=1).indices).sum().item()
                
                # model.train()
                # optimizer.zero_grad()
                # predict4 = model(back_augment_imgB_flow)
                # loss4 = loss_func(predict4, augment_label)
                # loss4.backward()
                # optimizer.step()

                # predict4 = torch.nn.Softmax(dim=1)(predict4)
                # total = total + augment_label.size(0)
                # correct = correct + (torch.max(predict4, dim=1).indices ==
                #                      torch.max(augment_label, dim=1).indices).sum().item()

                total_loss = total_loss + loss1 + loss2

                # print(f"        {label_type = }, {torch.max(augment_label, dim=1).indices = }")
                # print(f"        {predict1[0] = }")
                # print(f"        {torch.max(predict1, dim=1).indices = }, {loss1 = }")
                # print(f"        {predict2[0] = }")
                # print(f"        {torch.max(predict2, dim=1).indices = }, {loss2 = }")
                # print(f"        {predict3[0] = }")
                # print(f"        {torch.max(predict3, dim=1).indices = }, {loss3 = }")
                # print(f"        {predict4[0] = }")
                # print(f"        {torch.max(predict4, dim=1).indices = }, {loss4 = }")

                # torch.cuda.empty_cache()
                # break
            del normal_type
            del augment_flow_type
            del set1, set2
            del augment_imgA_flow, augment_imgB_flow
            # del back_augment_imgA_flow, back_augment_imgB_flow
            del augment_label
            del predict1, loss1
            del predict2, loss2
            # del predict3, loss3
            # del predict4, loss4
            del group_idx, imgA, imgA_depth, imgB, imgB_depth, flowAB, back_flowAB
            torch.cuda.empty_cache()
            if save_to_test and batch_idx == 4:
                break
            # break

        print(f"    train accuracy = {correct} / {total} = {correct / total}, {total_loss = }")
        del img0, img1, img2
        del img0_depth, img1_depth, img2_depth
        del flow01, flow12, flow02
        del back_flow01, back_flow12, back_flow02
        del total, correct
        del group
        del batch_idx, img_depth_flow
        torch.cuda.empty_cache()

        total = 0
        correct = 0
        confusion_matrix = np.zeros((num_classes, num_classes))
        for batch_idx, img_depth_flow in enumerate(tqdm(test_loader)):
            # img_depth_flow = img_depth_flow.type(torch.float32).to(device) 

            (img0, img1, img2, img0_depth, img1_depth, img2_depth,
            flow01, flow12, flow02, back_flow01, back_flow12, back_flow02) = get_data(img_depth_flow)


            batch_size = img0.size(0)
            group = [(img0, img0_depth, img1, img1_depth, flow01, back_flow01),
                     (img1, img1_depth, img2, img2_depth, flow12, back_flow12),
                     (img0, img0_depth, img2, img2_depth, flow02, back_flow02)]

            for group_idx, (imgA, imgA_depth, imgB, imgB_depth, flowAB, back_flowAB) in enumerate(group):
                imgA = imgA.type(torch.float32).to(device)
                imgB = imgB.type(torch.float32).to(device)
                imgA_depth = imgA_depth.type(torch.float32).to(device)
                imgB_depth = imgB_depth.type(torch.float32).to(device)
                flowAB = flowAB.type(torch.float32).to(device)
                back_flowAB = back_flowAB.type(torch.float32).to(device)
                    
                normal_type = utils.get_random(3, 0, False)

                augment_flow_type = utils.get_random(4, 0, False)
                if augment_flow_type >= 1.:
                    augment_flow_type = augment_flow_type + 4
                else:
                    augment_flow_type = normal_type

                fake_set1, fake_set2, _ = augment_flow(imgA, torch.ones_like(imgA_depth),
                            imgB, torch.ones_like(imgB_depth),
                            flowAB, back_flowAB, device=device, augment_flow_type=augment_flow_type)

                set1, set2, _ = augment_flow(imgA, imgA_depth, imgB, imgB_depth,
                            flowAB, back_flowAB, device=device, augment_flow_type=augment_flow_type)

                label_type = max(0, int(augment_flow_type) - 2) # 0, 1, 2 is normal
                augment_label = torch.zeros((batch_size, 1 + 5)).to(device)
                augment_label[:, label_type] = 1
                
                if save_to_test:
                    output_dir = f"testing/train_classifier/test_{label_type}_{epoch}_{batch_idx}_{group_idx}"
                    os.mkdir(output_dir)
                    augment_imgA_img0_depth = set1[1][0, 0].cpu().numpy()
                    augment_imgB_img0_depth = set1[1][0, 0].cpu().numpy()
                    plt.imsave(f"{output_dir}/augment_imgA_img0_depth.png", 1 / augment_imgA_img0_depth, cmap="magma")
                    plt.imsave(f"{output_dir}/augment_imgB_img0_depth.png", 1 / augment_imgB_img0_depth, cmap="magma")

                    fake_augment_imgA_img0 = fake_set1[0][0].permute(1, 2, 0).cpu().numpy()
                    fake_augment_imgB_img0 = fake_set2[0][0].permute(1, 2, 0).cpu().numpy()
                    fake_augment_imgA_img1 = fw(fake_set1[0], fake_set1[2], fake_set1[1])[0].permute(1, 2, 0).cpu().numpy() 
                    fake_augment_imgB_img1 = fw(fake_set2[0], fake_set2[2], fake_set2[1])[0].permute(1, 2, 0).cpu().numpy() 
                    cv2.imwrite(f"{output_dir}/augment_imgA_img0_fake.png", fake_augment_imgA_img0)
                    cv2.imwrite(f"{output_dir}/augment_imgB_img0_fake.png", fake_augment_imgB_img0)
                    cv2.imwrite(f"{output_dir}/augment_imgA_img1_fake.png", fake_augment_imgA_img1)
                    cv2.imwrite(f"{output_dir}/augment_imgB_img1_fake.png", fake_augment_imgB_img1)

                    fake_augment_imgA_flow_color = fake_set1[2][0:1].permute(0, 2, 3, 1).cpu()
                    fake_augment_imgB_flow_color = fake_set2[2][0:1].permute(0, 2, 3, 1).cpu()
                    fake_back_augment_imgA_flow_color = fake_set1[3][0:1].permute(0, 2, 3, 1).cpu()
                    fake_back_augment_imgB_flow_color = fake_set2[3][0:1].permute(0, 2, 3, 1).cpu()
                    _, fake_augment_imgA_flow_color = utils.color_flow(fake_augment_imgA_flow_color)
                    _, fake_augment_imgB_flow_color = utils.color_flow(fake_augment_imgB_flow_color)
                    _, fake_back_augment_imgA_flow_color = utils.color_flow(fake_back_augment_imgA_flow_color)
                    _, fake_back_augment_imgB_flow_color = utils.color_flow(fake_back_augment_imgB_flow_color)
                    cv2.imwrite(f"{output_dir}/augment_imgA_flow_fake.png", fake_augment_imgA_flow_color)
                    cv2.imwrite(f"{output_dir}/augment_imgB_flow_fake.png", fake_augment_imgB_flow_color)
                    cv2.imwrite(f"{output_dir}/back_augment_imgA_flow_fake.png", fake_back_augment_imgA_flow_color)
                    cv2.imwrite(f"{output_dir}/back_augment_imgB_flow_fake.png", fake_back_augment_imgB_flow_color)

                    augment_imgA_img0 = set1[0][0].permute(1, 2, 0).cpu().numpy()
                    augment_imgB_img0 = set2[0][0].permute(1, 2, 0).cpu().numpy()
                    augment_imgA_img1 = fw(set1[0], set1[2], set1[1])[0].permute(1, 2, 0).cpu().numpy() 
                    augment_imgB_img1 = fw(set2[0], set2[2], set2[1])[0].permute(1, 2, 0).cpu().numpy() 
                    cv2.imwrite(f"{output_dir}/augment_imgA_img0.png", augment_imgA_img0)
                    cv2.imwrite(f"{output_dir}/augment_imgB_img0.png", augment_imgB_img0)
                    cv2.imwrite(f"{output_dir}/augment_imgA_img1.png", augment_imgA_img1)
                    cv2.imwrite(f"{output_dir}/augment_imgB_img1.png", augment_imgB_img1)

                    augment_imgA_flow_color = set1[2][0:1].permute(0, 2, 3, 1).cpu()
                    augment_imgB_flow_color = set2[2][0:1].permute(0, 2, 3, 1).cpu()
                    back_augment_imgA_flow_color = set1[3][0:1].permute(0, 2, 3, 1).cpu()
                    back_augment_imgB_flow_color = set2[3][0:1].permute(0, 2, 3, 1).cpu()
                    _, augment_imgA_flow_color = utils.color_flow(augment_imgA_flow_color)
                    _, augment_imgB_flow_color = utils.color_flow(augment_imgB_flow_color)
                    _, back_augment_imgA_flow_color = utils.color_flow(back_augment_imgA_flow_color)
                    _, back_augment_imgB_flow_color = utils.color_flow(back_augment_imgB_flow_color)
                    cv2.imwrite(f"{output_dir}/augment_imgA_flow.png", augment_imgA_flow_color)
                    cv2.imwrite(f"{output_dir}/augment_imgB_flow.png", augment_imgB_flow_color)
                    cv2.imwrite(f"{output_dir}/back_augment_imgA_flow.png", back_augment_imgA_flow_color)
                    cv2.imwrite(f"{output_dir}/back_augment_imgB_flow.png", back_augment_imgB_flow_color)

                # augment_imgA_flow = resize(set1[2])
                # augment_imgB_flow = resize(set2[2])
                augment_imgA_flow = set1[2]
                augment_imgB_flow = set2[2]
                # augment_imgA_depth = set1[1]
                # augment_imgB_depth = set2[1]
                # back_augment_imgA_flow = set1[3]
                # back_augment_imgB_flow = set2[3]

                model.eval()
                with torch.no_grad():
                    predict1 = model(augment_imgA_flow)
                    # predict1 = model(torch.cat((augment_imgA_flow, augment_imgA_depth), axis=1))
                predict1 = torch.nn.Softmax(dim=1)(predict1)
                total = total + augment_label.size(0)
                correct = correct + (torch.max(predict1, dim=1).indices ==
                                     torch.max(augment_label, dim=1).indices).sum().item()

                for p, l in zip(torch.max(predict1, dim=1).indices, torch.max(augment_label, dim=1).indices):
                    confusion_matrix[p, l] = confusion_matrix[p, l] + 1

                model.eval()
                with torch.no_grad():
                    predict2 = model(augment_imgB_flow)
                    # predict2 = model(torch.cat((augment_imgB_flow, augment_imgB_depth), axis=1))
                predict2 = torch.nn.Softmax(dim=1)(predict2)
                total = total + augment_label.size(0)
                correct = correct + (torch.max(predict2, dim=1).indices ==
                                     torch.max(augment_label, dim=1).indices).sum().item()
                
                for p, l in zip(torch.max(predict2, dim=1).indices, torch.max(augment_label, dim=1).indices):
                    confusion_matrix[p, l] = confusion_matrix[p, l] + 1

                # model.eval()
                # with torch.no_grad():
                #     predict3 = model(back_augment_imgA_flow)
                # predict3 = torch.nn.Softmax(dim=1)(predict3)
                # total = total + augment_label.size(0)
                # correct = correct + (torch.max(predict3, dim=1).indices ==
                #                      torch.max(augment_label, dim=1).indices).sum().item()
                
                # model.eval()
                # with torch.no_grad():
                #     predict4 = model(back_augment_imgB_flow)
                # predict4 = torch.nn.Softmax(dim=1)(predict4)
                # total = total + augment_label.size(0)
                # correct = correct + (torch.max(predict4, dim=1).indices ==
                #                      torch.max(augment_label, dim=1).indices).sum().item()
                # print(f"        {augment_flow_type = }, {torch.max(augment_label, dim=1).indices = }")
                # print(f"        {predict1[0] = }")
                # print(f"        {torch.max(predict1, dim=1).indices = }")
                # print(f"        {predict2[0] = }")
                # print(f"        {torch.max(predict2, dim=1).indices = }")
                # print(f"        {predict3[0] = }")
                # print(f"        {torch.max(predict3, dim=1).indices = }")
                # print(f"        {predict4[0] = }")
                # print(f"        {torch.max(predict4, dim=1).indices = }")

                # torch.cuda.empty_cache()
                # break
            del normal_type
            del augment_flow_type
            del set1, set2
            del augment_imgA_flow, augment_imgB_flow
            # del back_augment_imgA_flow, back_augment_imgB_flow
            del augment_label
            del predict1
            del predict2
            # del predict3
            # del predict4
            del group_idx, imgA, imgA_depth, imgB, imgB_depth, flowAB, back_flowAB
            torch.cuda.empty_cache()
            # break
            if save_to_test and batch_idx == 4:
                break
        
        print(f"                    test accuracy = {correct} / {total} = {correct / total}")
        if correct / total >= best_accuracy:
            best_accuracy = correct / total
            output_dim = model.output_dim
            dropout = model.dropout
            Inencoder = model.use_dropout_in_encoder
            Inclassify = model.use_dropout_in_classify
            print(f"{confusion_matrix = }")
            torch.save(model.state_dict(), f"output/models/{output_dim=}{dropout=}{Inencoder=}{Inclassify=}.pt")
        del img0, img1, img2
        del img0_depth, img1_depth, img2_depth
        del flow01, flow12, flow02
        del back_flow01, back_flow12, back_flow02
        del total, correct
        del group
        del batch_idx, img_depth_flow
        torch.cuda.empty_cache()

        lr = max(0.0000002, lr - 0.0000004)

        if save_to_test and epoch == 4:
            break


    nvidia_smi.nvmlShutdown()
