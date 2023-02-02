import torch
import data


def test_flow_range():
    # val_dataset = data.KITTI(split="training")
    val_dataset = data.RAFTAugmentedReDWeb(split="training")

    s0_10, s10_40, s40_90, s90_160, s160plus = 0, 0, 0, 0, 0
    for val_id in range(len(val_dataset)):
        # img1_gt, img2_gt, flow_gt, valid_gt = val_dataset[val_id]
        img1_gt, img2_gt, flow_gt, depth_gt, valid_gt, label = val_dataset[val_id]

        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        valid = torch.sum(valid_gt >= 0.5)
        s0_10 = s0_10 + torch.sum((mag < 10) * (valid_gt >= 0.5))
        s10_40 = s10_40 + torch.sum((mag >= 10) * (mag < 40) * (valid_gt >= 0.5))
        s40_90 = s40_90 + torch.sum((mag >= 40) * (mag < 90) * (valid_gt >= 0.5))
        s90_160 = s90_160 + torch.sum((mag >= 90) * (mag < 160) * (valid_gt >= 0.5))
        s160plus = s160plus + torch.sum((mag >= 160) * (valid >= 0.5))

        _all = s0_10 + s10_40 + s40_90 + s90_160 + s160plus
        print(f"{s0_10 = }, {s10_40 = }, {s40_90 = }, {s90_160 = }, {s160plus = }")
        print(f"s0_10 ratio = {s0_10 / _all}, s10_40 ratio = {s10_40 / _all}, s40_90 ratio = {s40_90 / _all}, s90_160 ratio = {s90_160 / _all}, s160plus ratio = {s160plus / _all}")
    pass


if __name__ == "__main__":
    test_flow_range()
