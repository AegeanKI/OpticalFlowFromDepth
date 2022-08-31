from classifier import Classifier
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda import is_available
from torch import device, from_numpy
import torch
from dataloader import dCOCOLoader
from augment import augment_flow
import numpy as np

def get_loaders(batch_size):
    train_dataset = dCOCOLoader("train")
    # test_dataset = dCOCOLoader("test")
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    return train_loader, None
    # return train_loader, test_loader
    # return train_dataset, test_dataset

def get_cuda_device():
    return device("cuda" if is_available() else "cpu")

if __name__ == "__main__":
    lr = 0.1
    batch_size = 1
    size = (384, 384) 
    max_epoch = 10
    # model = Classifier(size, use_small=True).double()
    model = Classifier(size, use_small=False).double()
    # model.load_state_dict(torch.load("output/models/classifier_without_softmax.pt"))
    optimizer = Adam(model.parameters(), lr=lr)
    loss_function = CrossEntropyLoss()
    device = get_cuda_device()

    train_loader, test_loader = get_loaders(batch_size)

    count = 0
    correct = 0
    for epoch in range(max_epoch):
        for idx, (img0, img1, img2, img0_depth, img1_depth, img2_depth, flow01, flow12, flow02, back_flow01, back_flow12, back_flow02) in enumerate(train_loader):
            img0 = img0.squeeze().permute(1, 2, 0).detach().numpy()
            img1 = img1.squeeze().permute(1, 2, 0).detach().numpy()
            img2 = img2.squeeze().permute(1, 2, 0).detach().numpy()
            img0_depth = img0_depth.squeeze().detach().numpy()
            img1_depth = img1_depth.squeeze().detach().numpy()
            img2_depth = img2_depth.squeeze().detach().numpy()
            flow01 = flow01.permute(0, 2, 3, 1)
            flow12 = flow12.permute(0, 2, 3, 1)
            flow02 = flow02.permute(0, 2, 3, 1)
            back_flow01 = back_flow01.permute(0, 2, 3, 1)
            back_flow12 = back_flow12.permute(0, 2, 3, 1)
            back_flow02 = back_flow02.permute(0, 2, 3, 1)
            # print(f"{img0.shape = }")
            # print(f"{img1_depth.shape = }")
            # print(f"{flow02.shape = }")

            group = [(img0, img0_depth, img1, img1_depth, flow01, back_flow01),
                     (img1, img1_depth, img2, img2_depth, flow12, back_flow12),
                     (img0, img0_depth, img2, img2_depth, flow02, back_flow02)]


            for group_idx, (imgA, imgA_depth, imgB, imgB_depth, flowAB, back_flowAB) in enumerate(group):
                print(f"{epoch = }, {idx} / {len(train_loader)} - {group_idx} / 3", end='')
                set1, set2, augment_flow_type = augment_flow(imgA, imgA_depth, imgB, imgB_depth,
                                                             flowAB, back_flowAB, size)
                augment_label = torch.zeros((1, 1 + 5))
                augment_label[0, max(0, augment_flow_type - 2)] = 1
                print(f", {augment_flow_type = }", end='')

                predict1 = model(set1[3])
                optimizer.zero_grad()
                loss = loss_function(predict1, augment_label)
                loss.backward()
                optimizer.step()
                correct = correct + (torch.argmax(predict1) == max(0, augment_flow_type - 2)) 
                count = count + 1
                print(f", {predict1 = }", end='')

                predict2 = model(set2[3])
                optimizer.zero_grad()
                loss = loss_function(predict2, augment_label)
                loss.backward()
                optimizer.step()
                correct = correct + (torch.argmax(predict2) == max(0, augment_flow_type - 2)) 
                count = count + 1
                print(f", {predict2 = }", end='')

                label = torch.zeros((1, 1 + 5))
                label[0, 0] = 1
                predict3 = model(flowAB)
                optimizer.zero_grad()
                loss = loss_function(predict3, label)
                loss.backward()
                optimizer.step()
                correct = correct + (torch.argmax(predict3) == 0) 
                count = count + 1
                print(f", {predict3 = }", end='')

                print(f", acc = {correct} / {count} = {correct / count}")

            if idx % 5 == 4:
                torch.save(model.state_dict(), "output/models/classifier.pt")
