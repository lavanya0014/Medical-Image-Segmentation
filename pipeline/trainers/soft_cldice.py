"""
Soft clDice loss (differentiable centerline-Dice), for the C3 "combined KD + optional
clDice" variant. Standard soft-skeletonization approach (Shit et al., 2021).
"""
import torch
import torch.nn.functional as F


def soft_erode(img):
    if img.dim() == 5:  # 3D: (B, C, D, H, W)
        p1 = -F.max_pool3d(-img, (3, 1, 1), 1, (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), 1, (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), 1, (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    else:  # 2D
        p1 = -F.max_pool2d(-img, (3, 1), 1, (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), 1, (0, 1))
        return torch.min(p1, p2)


def soft_dilate(img):
    if img.dim() == 5:
        return F.max_pool3d(img, 3, 1, 1)
    return F.max_pool2d(img, 3, 1, 1)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, iters=10):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(pred_prob, target_onehot, iters=10, smooth=1.0):
    """
    pred_prob: (B, C, ...) softmax probabilities, foreground channel(s) only used
    target_onehot: (B, C, ...) one-hot ground truth, same shape
    Uses the foreground (channel 1) for binary vessel segmentation.
    """
    pred_fg = pred_prob[:, 1:2]
    gt_fg = target_onehot[:, 1:2].float()

    skel_pred = soft_skeletonize(pred_fg, iters)
    skel_true = soft_skeletonize(gt_fg, iters)

    tprec = (torch.sum(skel_pred * gt_fg) + smooth) / (torch.sum(skel_pred) + smooth)
    tsens = (torch.sum(skel_true * pred_fg) + smooth) / (torch.sum(skel_true) + smooth)
    cl_dice = 1 - 2 * (tprec * tsens) / (tprec + tsens + 1e-8)
    return cl_dice
