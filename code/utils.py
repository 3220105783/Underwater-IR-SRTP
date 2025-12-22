# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------- 辅助函数：计算IoU（交并比）--------------------------
def calculate_iou(pred, target, smooth=1e-6):
    """
    计算批量图片的IoU（交并比）
    :param pred: 模型输出的logits（未经过sigmoid），shape: (batch_size, 1, H, W)
    :param target: 真实标签，shape: (batch_size, 1, H, W)
    :param smooth: 平滑项，避免分母为0
    :return: 批量平均IoU值
    """
    # 将logits转为概率并二值化（与train.py保持一致，阈值0.3）
    pred = torch.sigmoid(pred) > 0.3
    pred = pred.float()  # 转为float类型，便于计算

    # 展平为一维向量（batch_size * H * W）
    pred = pred.view(-1)
    target = target.view(-1)

    # 计算交集和并集
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    # 计算IoU（添加平滑项避免除零）
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


# -------------------------- 损失函数：Focal Loss（解决小目标/样本不平衡）--------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        """
        :param gamma: 聚焦参数，gamma越大，对难样本的权重越大（默认2，适合小目标）
        :param alpha: 类别权重（默认None，自动适配二分类）
        :param reduction: 损失聚合方式（mean/sum/none）
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        # 计算基础的交叉熵损失（未经过softmax/sigmoid，因为input是logits）
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        # 计算聚焦因子：pt = exp(-ce_loss)，(1-pt)^gamma 让难样本权重增大
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 应用类别权重（正样本-裂缝权重10，负样本-背景权重1）
        if self.alpha is not None:
            # 确保alpha是tensor且与target设备一致
            alpha = self.alpha.to(target.device)
            alpha_t = alpha[target.long()]  # 按标签选择对应权重
            focal_loss = alpha_t * focal_loss

        # 聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# -------------------------- 损失函数：Dice Loss（解决小目标分割）--------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        :param smooth: 平滑项，避免分母为0（默认1e-6）
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred是logits，先转为sigmoid概率
        pred = torch.sigmoid(pred)

        # 展平为一维向量
        pred = pred.view(-1)
        target = target.view(-1)

        # 计算交集和并集
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        # 计算Dice系数（1-Dice作为损失）
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


# -------------------------- 组合损失：FocalDiceLoss（小目标最优解）--------------------------
class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.4, dice_weight=0.6):
        """
        :param focal_weight: Focal Loss的权重（默认0.4）
        :param dice_weight: Dice Loss的权重（默认0.6，增强小目标关注）
        """
        super(FocalDiceLoss, self).__init__()
        # 初始化Focal Loss（alpha=[背景权重, 裂缝权重]，裂缝权重50）
        self.focal = FocalLoss(gamma=2, alpha=torch.tensor([1.0, 50.0]))
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        # 计算两个损失并加权求和
        focal_loss = self.focal(pred, target)
        dice_loss = self.dice(pred, target)
        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return total_loss


# -------------------------- 兼容旧版：BCEDiceLoss（可选，如需回退使用）--------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.7):
        super(BCEDiceLoss, self).__init__()
        # BCE带类别权重（裂缝权重10）
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss