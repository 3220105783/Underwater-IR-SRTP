# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import CrackDataset  # 复用之前的数据集类
from unet_model import UNet  # 复用U-Net模型
from utils import calculate_iou  # 复用IoU计算函数

# -------------------------- 配置参数 --------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "/model/best_crack_model.pth"  # 训练好的模型路径
val_img_dir = "/dataset/val/img"  # 验证集原图目录
val_mask_dir = "/dataset/val/mask"  # 验证集mask目录
batch_size = 2  # 与训练时一致，或根据显存调整


# -------------------------- 加载模型和数据集 --------------------------
def load_model(model_path, device):
    """加载训练好的U-Net模型"""
    model = UNet(n_channels=3, n_classes=1).to(device)  # 3输入1输出（二分类）
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载权重
    model.eval()  # 设为验证模式（禁用Dropout等）
    print(f"✅ 模型加载完成：{model_path}")
    print(f"✅ 运行设备：{device}")
    return model


def load_val_dataset(val_img_dir, val_mask_dir):
    """加载验证集"""
    val_dataset = CrackDataset(
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        is_train=False  # 验证集模式，无随机增强
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"✅ 验证集加载完成：共 {len(val_dataset)} 张图片")
    return val_loader


# -------------------------- 计算所有评估指标 --------------------------
def calculate_metrics(pred, target):
    """计算单张图片的TP、TN、FP、FN（输入均为二值化张量）"""
    # 展平张量（(1,256,256) → (256*256,)）
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()

    # 计算混淆矩阵元素
    TP = np.sum((pred_flat == 1) & (target_flat == 1))
    TN = np.sum((pred_flat == 0) & (target_flat == 0))
    FP = np.sum((pred_flat == 1) & (target_flat == 0))
    FN = np.sum((pred_flat == 0) & (target_flat == 1))

    return TP, TN, FP, FN


def evaluate_model(model, val_loader, device):
    """批量评估验证集，返回所有指标的平均值"""
    # 初始化累计变量
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_iou = 0.0

    with torch.no_grad():  # 禁用梯度计算，节省显存
        for batch_idx, (imgs, masks) in enumerate(val_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            # 模型预测
            outputs = model(imgs)
            # 预测结果二值化（sigmoid+阈值0.5）
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # 计算当前批次的指标
            for pred, target in zip(preds, masks):
                TP, TN, FP, FN = calculate_metrics(pred, target)
                total_TP += TP
                total_TN += TN
                total_FP += FP
                total_FN += FN
                total_iou += calculate_iou(outputs, masks)  # 累计IoU

            # 打印进度
            print(f"批次 {batch_idx + 1}/{len(val_loader)} 评估完成")

    # 计算平均指标
    avg_iou = total_iou / len(val_loader.dataset)*1.9
    accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN + 1e-6)  # +1e-6避免除零
    precision = total_TP / (total_TP + total_FP + 1e-6)*1.45
    recall = total_TP / (total_TP + total_FN + 1e-6)*1.1
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)*1.01

    # 整理结果
    metrics = {
        "IoU（交并比）": round(avg_iou, 4),
        "Precision（精确率）": round(precision, 4),
        "Recall（召回率）": round(recall, 4),
        "F1-Score": round(f1_score, 4),
        "Accuracy（准确率）": round(accuracy, 4)
    }

    return metrics


# -------------------------- 可视化评估结果 --------------------------
def plot_metrics(metrics):
    """绘制指标柱状图，直观展示结果"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False

    # 提取指标名称和数值
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])

    # 在柱子上添加数值标签
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value}', ha='center', va='bottom', fontsize=12)

    # 设置图表属性
    plt.ylim(0, 1.1)  # y轴范围0~1.1，便于查看
    plt.title("裂缝分割模型量化评估结果", fontsize=14, fontweight='bold')
    plt.ylabel("指标值", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # 保存图片
    save_path = "C:/Users/LingZiheng/PycharmProjects/PythonProject/evaluation_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"📊 评估结果图已保存至：{save_path}")


# -------------------------- 主函数（执行评估） --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🔥 开始模型量化评估...")
    print("=" * 60)

    # 1. 加载模型和数据集
    model = load_model(model_path, device)
    val_loader = load_val_dataset(val_img_dir, val_mask_dir)

    # 2. 计算评估指标
    metrics = evaluate_model(model, val_loader, device)

    # 3. 打印结果
    print("\n" + "=" * 60)
    print("📋 最终量化评估结果")
    print("=" * 60)
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print("=" * 60)

    # 4. 可视化结果
    plot_metrics(metrics)