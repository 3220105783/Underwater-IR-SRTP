import json
import os
from PIL import Image
import numpy as np

# 配置路径（根据你的实际路径修改）
IMAGE_DIR = "C:/Users/LingZiheng/PycharmProjects/PythonProject/dataset/images"  # 原图和json文件所在目录
MASK_DIR = "C:/Users/LingZiheng/PycharmProjects/PythonProject/dataset/masks"  # 生成的mask图保存目录

# 创建mask保存目录
if not os.path.exists(MASK_DIR):
    os.makedirs(MASK_DIR)

# 遍历所有json文件
for filename in os.listdir(IMAGE_DIR):
    if filename.endswith(".json"):
        json_path = os.path.join(IMAGE_DIR, filename)
        img_name = filename.replace(".json", "")  # 原图文件名（无后缀）

        # 读取json文件
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 获取原图尺寸
        img_height = data["imageHeight"]
        img_width = data["imageWidth"]

        # 创建空白mask（背景为0）
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # 解析标注的裂缝区域（类别为crack）
        for shape in data["shapes"]:
            if shape["label"] == "crack":
                # 获取多边形顶点
                points = np.array(shape["points"], dtype=np.int32)
                # 填充多边形（裂缝区域设为255）
                from PIL import ImageDraw

                img = Image.new("L", (img_width, img_height), 0)
                ImageDraw.Draw(img).polygon(tuple(map(tuple, points)), fill=255)
                mask += np.array(img)

        # 保存mask图
        mask_img = Image.fromarray(mask)
        mask_path = os.path.join(MASK_DIR, f"{img_name}_mask.png")
        mask_img.save(mask_path)
        print(f"已生成mask：{mask_path}")

print("所有mask生成完成！")