import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from unet_model import UNet
from dataset import val_transform  # 复用验证集的预处理

# 配置路径
model_path = "../model/best_unet_crack.pth"  # 训练好的模型路径
test_img_dir = "../dataset/val/img"  # 测试图片目录（用验证集测试）
save_result_dir = "../predict_results"  # 预测结果保存目录
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建结果保存目录
if not os.path.exists(save_result_dir):
    os.makedirs(save_result_dir)

# 加载模型
model = UNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()  # 切换到评估模式


# 预测函数
def predict_image(img_path):
    # 读取图片
    image = Image.open(img_path).convert("RGB")
    img_np = np.array(image)

    # 预处理（与训练时一致）
    transform = val_transform
    augmented = transform(image=img_np)
    img_tensor = augmented["image"].unsqueeze(0).to(device)  # 增加batch维度

    # 预测
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.sigmoid(output) > 0.5  # 二值化（阈值0.5）
        pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()  # 去除多余维度

    return img_np, pred_mask


# 批量预测并可视化
test_img_names = [f for f in os.listdir(test_img_dir) if f.endswith((".jpg", ".png"))]
for img_name in test_img_names:
    img_path = os.path.join(test_img_dir, img_name)
    img_np, pred_mask = predict_image(img_path)

    # 读取真实mask（用于对比）
    mask_name = img_name.replace(".jpg", "_mask.png").replace(".png", "_mask.png")
    mask_path = os.path.join("../dataset/val/mask", mask_name)
    true_mask = np.array(Image.open(mask_path).convert("L")) / 255.0  # 归一化到0-1

    # 可视化（原图 + 真实mask + 预测mask）
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask, cmap="gray")
    plt.title("True Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    # 保存结果
    save_path = os.path.join(save_result_dir, f"result_{img_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"预测结果已保存：{save_path}")

print("所有图片预测完成！结果保存在：", save_result_dir)