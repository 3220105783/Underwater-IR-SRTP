# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import CrackDataset
from unet_model import UNet
from utils import FocalDiceLoss, calculate_iou

# -------------------------- 配置参数（固定学习率，无调度）--------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1  # 按显存调整，RTX2050 4GB建议设为1
epochs = 60  # 总训练轮次（可按需修改）
fixed_lr = 1e-4  # 固定学习率（无需调整）
patience = 20  # 早停耐心值（连续8轮IoU无提升则停止）
model_save_path = "C:/Users/LingZiheng/PycharmProjects/PythonProject/model/best_crack_model.pth"

# 数据集路径（修改为你的实际路径）
train_img_dir = "C:/Users/LingZiheng/PycharmProjects/PythonProject/dataset/train/img"
train_mask_dir = "C:/Users/LingZiheng/PycharmProjects/PythonProject/dataset/train/mask"
val_img_dir = "C:/Users/LingZiheng/PycharmProjects/PythonProject/dataset/val/img"
val_mask_dir = "C:/Users/LingZiheng/PycharmProjects/PythonProject/dataset/val/mask"


# -------------------------- 数据加载 --------------------------
def load_data():
    print("\n📥 开始加载数据集...")
    # 训练集（无过采样，如需开启设oversample=True）
    train_dataset = CrackDataset(
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        is_train=True,
        oversample=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # 验证集
    val_dataset = CrackDataset(
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        is_train=False,
        oversample=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"\n📊 数据加载完成：")
    print(f"   - 训练集：{len(train_dataset)} 样本，{len(train_loader)} 批次")
    print(f"   - 验证集：{len(val_dataset)} 样本，{len(val_loader)} 批次")

    # 无有效样本时报错
    if len(train_dataset) == 0:
        raise ValueError("❌ 训练集无有效样本！请检查文件名匹配和文件路径")
    if len(val_dataset) == 0:
        raise ValueError("❌ 验证集无有效样本！请检查文件名匹配和文件路径")

    return train_loader, val_loader


# -------------------------- 训练一轮 --------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="训练")

    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)

        # 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, masks)

        # 反向传播+优化（固定学习率，无调度）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计损失
        total_loss += loss.item() * imgs.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{fixed_lr:.6f}"})

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


# -------------------------- 验证一轮 --------------------------
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    pbar = tqdm(loader, desc="验证")

    with torch.no_grad():
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            # 二值化（阈值0.3，适配浅灰裂缝）
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # 计算IoU
            batch_iou = 0.0
            for output, target in zip(outputs, masks):  # 遍历原始输出outputs和目标mask
                iou = calculate_iou(output.unsqueeze(0), target.unsqueeze(0))  # 补充通道维度以匹配函数要求
                batch_iou += iou
            batch_avg_iou = batch_iou / len(imgs)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "IoU": f"{batch_avg_iou:.4f}"})

            total_loss += loss.item() * imgs.size(0)
            total_iou += batch_avg_iou * len(imgs)

    avg_loss = total_loss / len(loader.dataset)
    avg_iou = total_iou / len(loader.dataset)
    return avg_loss, avg_iou


# -------------------------- 训练可视化 --------------------------
def plot_training_history(train_losses, val_losses, val_ious):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 损失曲线
    ax1.plot(train_losses, label="训练损失", color="#e74c3c")
    ax1.plot(val_losses, label="验证损失", color="#3498db")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("损失值")
    ax1.set_title("训练/验证损失曲线（固定学习率）")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # IoU曲线
    ax2.plot(val_ious, label="验证IoU", color="#2ecc71", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU值")
    ax2.set_title("验证IoU曲线")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("training_history_fixed_lr.png", dpi=150, bbox_inches="tight")
    plt.show()


# -------------------------- 主训练流程 --------------------------
def main():
    print("=" * 60)
    print("🔥 开始裂缝分割模型训练（固定学习率版）")
    print("=" * 60)
    print(f"📌 配置信息：")
    print(f"   - 设备：{device}")
    print(f"   - 批次大小：{batch_size}")
    print(f"   - 总轮次：{epochs}")
    print(f"   - 固定学习率：{fixed_lr:.6f}")
    print("=" * 60)

    # 1. 加载数据
    train_loader, val_loader = load_data()

    # 2. 初始化模型、损失函数、优化器（无学习率调度器）
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = FocalDiceLoss(focal_weight=0.4, dice_weight=0.6).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=fixed_lr,  # 直接使用固定学习率
        weight_decay=1e-4  # 权重衰减，防止过拟合
    )

    # 3. 训练记录与早停初始化
    train_losses = []
    val_losses = []
    val_ious = []
    best_iou = 0.0
    patience_counter = 0

    # 4. 开始训练（无学习率调整）
    for epoch in range(epochs):
        print(f"\n📌 Epoch {epoch + 1}/{epochs}")
        print(f"   当前学习率：{fixed_lr:.6f}（固定不变）")
        print("-" * 40)

        # 训练与验证
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        # 记录历史数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        # 打印本轮结果
        print(f"📊 Epoch {epoch + 1} 结果：")
        print(f"   - 训练损失：{train_loss:.4f}")
        print(f"   - 验证损失：{val_loss:.4f}")
        print(f"   - 验证IoU：{val_iou:.4f}")

        # 早停机制+保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), model_save_path)
            print(f"🏆 保存最佳模型（IoU：{best_iou:.4f}）")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⚠️  早停计数器：{patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"🛑 连续{patience}轮IoU无提升，提前停止训练")
                break

    # 5. 训练完成可视化
    print(f"\n" + "=" * 60)
    print(f"🎉 训练结束！最佳验证IoU：{best_iou:.4f}")
    print(f"📁 最佳模型保存路径：{model_save_path}")
    print("=" * 60)
    plot_training_history(train_losses, val_losses, val_ious)


if __name__ == "__main__":
    main()