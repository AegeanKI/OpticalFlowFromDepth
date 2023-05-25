from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets

from torch.utils.tensorboard import SummaryWriter

from my_classifier import RAFTClassifier, Classifier
from torch.nn import CrossEntropyLoss
import json

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    torch.cuda.set_device(args.gpus[0])

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs' and not args.is_first_stage:
        model.module.freeze_bn()
    # if args.stage != 'chairs' and args.stage != 'augmentedredweb' and args.stage != 'augmenteddiml':
    #     model.module.freeze_bn()

    # =====
    if args.add_classifier:
        classifier_checkpoints_dir = f"outputs/models/{args.classifier_checkpoint_timestamp}"
        train_acc = args.classifier_checkpoint_train_acc
        test_acc = args.classifier_checkpoint_test_acc
        classifier_checkpoints_name = f"{train_acc=}_{test_acc=}.pt"
        with open(f"{classifier_checkpoints_dir}/args.txt") as f:
            classifier_args = json.load(f)
        print(f"{classifier_args = }")
        if not classifier_args["use_depth_in_classifier"]:
            classifier = Classifier(device=args.gpus[0],
                                    output_dim=classifier_args["output_dim"],
                                    dropout=classifier_args["dropout"],
                                    use_small=classifier_args["use_small"],
                                    use_dropout_in_encoder=classifier_args["use_dropout_in_encoder"],
                                    use_dropout_in_classify=classifier_args["use_dropout_in_classify"],
                                    use_average_pooling=classifier_args["use_average_pooling"])
        else:
            classifier = RAFTClassifier(device=args.gpus[0],
                                        output_dim=classifier_args["output_dim"],
                                        dropout=classifier_args["dropout"],
                                        use_small=classifier_args["use_small"],
                                        use_dropout_in_encoder=classifier_args["use_dropout_in_encoder"],
                                        use_dropout_in_classify=classifier_args["use_dropout_in_classify"],
                                        use_average_pooling=classifier_args["use_average_pooling"])
        classifier.load_state_dict(torch.load(f"{classifier_checkpoints_dir}/{classifier_checkpoints_name}",
                                              map_location=f"cuda:{args.gpus[0]}"))
        classifier.to(args.gpus[0])
        classifier.eval()
        classify_loss_func = CrossEntropyLoss()

        classify_loss_weight = args.classify_loss_weight_init

        h, w = args.original_image_size
    # =====
    if args.add_forward_backward:
        forward_backward_loss_weight = args.forward_backward_loss_weight_init

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = args.val_freq
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, back_flow, image1_depth, image2_depth, valid, back_valid, label = [x.cuda() for x in data_blob]
            # print(f"{image1.shape = }")
            # print(f"{flow.shape = }")
            # print(f"{valid.shape = }")
            # print(f"{image1_depth.shape = }")

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)
            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)

            print(f"{total_steps}: loss = {loss.item():.10f}", end='')
            # =====
            if args.add_classifier:
                if ("not_normalize_dataset" in classifier_args) and (not classifier_args["not_normalize_dataset"]):
                    normalized_predicted_flow = flow_predictions[-1]
                    normalized_predicted_flow[:, 0] = normalized_predicted_flow[:, 0] / h
                    normalized_predicted_flow[:, 1] = normalized_predicted_flow[:, 1] / w
                    normalized_predicted_flow = normalized_predicted_flow.float()
                    normalized_depth = (image1_depth / 100).float()
                    normalized_predicted_flow_depth = torch.cat((normalized_predicted_flow, normalized_depth), axis=1)

                    if not classifier_args["use_depth_in_classifier"]:
                        predict1 = classifier(normalized_predicted_flow)
                    else:
                        predict1 = classifier(normalized_predicted_flow_depth)
                else:
                    predicted_flow = flow_predictions[-1].float()
                    depth = image1_depth.float()
                    predicted_flow_depth = torch.cat((predicted_flow, depth), axis=1)

                    if not classifier_args["use_depth_in_classifier"]:
                        predict1 = classifier(predicted_flow)
                    else:
                        predict1 = classifier(predicted_flow_depth)


                classify_loss = classify_loss_func(predict1, label)
                loss = loss + classify_loss * classify_loss_weight
            # ====

            if args.add_forward_backward:
                if args.use_ground_truth_backward:
                    predicted_back_flow = back_flow.detach()
                else:
                    with torch.no_grad():
                        back_flow_predictions = model(image2.detach(), image1.detach(), iters=args.iters)
                    predicted_back_flow = back_flow_predictions[-1].detach()
                predicted_flow = flow_predictions[-1].detach()
                # predicted_back_flow = back_flow.detach().cuda() # gt
                # predicted_flow = flow.detach().cuda()           # gt
                B, C, H, W = predicted_back_flow.size()
                # mesh grid 
                xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
                yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
                xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
                yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
                grid = torch.cat((xx, yy), 1).float()

                grid = grid.cuda() + predicted_flow

                grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0 
                grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

                grid = grid.permute(0, 2, 3, 1)
                warped_predicted_back_flow = F.grid_sample(predicted_back_flow, grid, padding_mode='border')
                backward_valid_mask = back_valid.detach().view(B, 1, H, W).repeat(1, 2, 1, 1)
                forward_valid_mask = valid.detach().view(B, 1, H, W).repeat(1, 2, 1, 1)
                backward_valid_mask = F.grid_sample(backward_valid_mask, grid, padding_mode='zeros')

                backward_valid_mask[backward_valid_mask < 0.9999] = 0
                backward_valid_mask[backward_valid_mask > 0] = 1

                full_mask = forward_valid_mask * backward_valid_mask 
                forward_backward_loss = torch.mean(torch.abs((predicted_flow + warped_predicted_back_flow) * full_mask))
                loss = loss + forward_backward_loss * forward_backward_loss_weight
                # loss = loss + forward_backward_loss.detach()
                
            if args.add_classifier:
                print(f" + {classify_loss.item():.10f} * {classify_loss_weight:.3f}", end='')
                classify_loss_weight = classify_loss_weight + args.classify_loss_weight_increase
                classify_loss_weight = max(args.min_classify_loss_weight, classify_loss_weight)
                classify_loss_weight = min(args.max_classify_loss_weight, classify_loss_weight)
            if args.add_forward_backward:
                print(f" + {forward_backward_loss.item():.10f} * {forward_backward_loss_weight:.3f}", end='')
                forward_backward_loss_weight = forward_backward_loss_weight + args.forward_backward_loss_weight_increase
                forward_backward_loss_weight = max(args.min_forward_backward_loss_weight, forward_backward_loss_weight)
                forward_backward_loss_weight = min(args.max_forward_backward_loss_weight, forward_backward_loss_weight)
            print(f" = {loss.item():.10f}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))
                    elif val_dataset == 'kitti12':
                        results.update(evaluate.validate_kitti12(model.module))
                    elif val_dataset == 'finetunekitti':
                        results.update(evaluate.validate_finetunekitti15(model.module))
                        results.update(evaluate.validate_kitti12(model.module))

                logger.write_dict(results)
                
                model.train()
                # if args.stage != 'chairs' and args.stage != 'augmentedredweb' and args.stage != 'augmenteddiml':
                #     model.module.freeze_bn()
                if args.stage != 'chairs' and not args.is_first_stage:
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps or total_steps > args.early_stop:
                should_keep_training = False
                break

            del image1, image2, image1_depth, image2_depth, valid, back_valid, flow, back_flow
            if args.add_noise:
                del stdv
            del flow_predictions
            del metrics

            if args.add_classifier:
                if ("not_normalize_dataset" in classifier_args) and (not classifier_args["not_normalize_dataset"]):
                    del normalized_predicted_flow
                    del normalized_depth
                    del normalized_predicted_flow_depth
                else:
                    del predicted_flow
                    del depth
                    del predicted_flow_depth
                del predict1
                del classify_loss
                
                classify_loss_weight = classify_loss_weight + args.classify_loss_weight_increase
                classify_loss_weight = max(args.min_classify_loss_weight, classify_loss_weight)
                classify_loss_weight = min(args.max_classify_loss_weight, classify_loss_weight)
            # ====

            if args.add_forward_backward:
                model.eval()
                if not args.use_ground_truth_backward:
                    del back_flow_predictions
                del predicted_back_flow
                del B, C, H, W
                del xx, yy
                del grid
                del warped_predicted_back_flow
                del forward_valid_mask, backward_valid_mask
                del full_mask
                del forward_backward_loss
            del loss
            del i_batch, data_blob
        #     break
        # break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--original_image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')

    parser.add_argument('--val_freq', type=int, default=5000)
    parser.add_argument('--add_classifier', action='store_true')
    parser.add_argument('--classifier_checkpoint_timestamp')
    parser.add_argument('--classifier_checkpoint_train_acc', type=float)
    parser.add_argument('--classifier_checkpoint_test_acc', type=float)
    parser.add_argument('--classify_loss_weight_init', type=float, default=1)
    parser.add_argument('--classify_loss_weight_increase', type=float, default=-0.00002)
    parser.add_argument('--max_classify_loss_weight', type=float, default=1)
    parser.add_argument('--min_classify_loss_weight', type=float, default=0)
    parser.add_argument('--add_forward_backward', action='store_true')
    parser.add_argument('--forward_backward_loss_weight_init', type=float, default=1)
    parser.add_argument('--forward_backward_loss_weight_increase', type=float, default=-0.00002)
    parser.add_argument('--max_forward_backward_loss_weight', type=float, default=1)
    parser.add_argument('--min_forward_backward_loss_weight', type=float, default=0)
    parser.add_argument('--early_stop', default=1e9, type=int)
    parser.add_argument('--is_first_stage', action='store_true')
    parser.add_argument('--use_ground_truth_backward', action='store_true')

    args = parser.parse_args()

    # if args.add_forward_backward:
    #     args.lr = args.lr / 2

    seed = 1234

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    try:
        random.seed(seed)
    except:
        pass
    try:
        loader.sampler.generator.manual_seed(seed)
    except:
        pass

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
