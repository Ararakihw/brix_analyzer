import os
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# 统一特征长度
TARGET_FEATURE_LENGTH = 1600  # 固定到1600维

# 提取光谱特征（统一长度）
def extract_spectrum_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = np.mean(gray, axis=0)  # 原始特征（不同图片可能长度不同）

    # 重新采样到统一长度
    x_old = np.linspace(0, 1, len(feature))
    f = interp1d(x_old, feature, kind='linear')
    x_new = np.linspace(0, 1, TARGET_FEATURE_LENGTH)
    feature_resampled = f(x_new)
    return feature_resampled

# 提取有糖度的光谱图数据并保存到CSV
def process_training_data(train_dir, output_csv):
    features = []
    labels = []
    filenames = []

    for filename in os.listdir(train_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            try:
                brix = float(filename.split("_")[0])  # 假设文件名格式为糖度值_图片名
                img_path = os.path.join(train_dir, filename)
                spectrum_feature = extract_spectrum_features(img_path)
                features.append(spectrum_feature)
                labels.append(brix)
                filenames.append(filename)
            except Exception as e:
                print(f"⚠️ 跳过无法识别的文件名: {filename}")

    # 转成numpy数组
    X = np.vstack(features)  # 每张图都有一样多的特征了！
    y = np.array(labels)

    # 保存到CSV文件
    csv_data = pd.DataFrame(X)
    csv_data['糖度(Brix)'] = labels
    csv_data.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 训练数据已保存到 {output_csv}")

# 提取没有糖度的光谱图数据并保存到CSV
def process_test_data(test_dir, output_csv):
    features = []
    filenames = []

    for filename in os.listdir(test_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            try:
                img_path = os.path.join(test_dir, filename)
                spectrum_feature = extract_spectrum_features(img_path)
                features.append(spectrum_feature)
                filenames.append(filename)
            except Exception as e:
                print(f"⚠️ 跳过无法识别的文件名: {filename}")

    # 转成numpy数组
    X = np.vstack(features)

    # 保存到CSV文件
    csv_data = pd.DataFrame(X)
    csv_data.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 预测数据已保存到 {output_csv}")

# 调用函数处理数据
train_dir = "train/other"  # 训练数据文件夹
test_dir = "test"    # 预测数据文件夹
process_training_data(train_dir, "data/train_data.csv")  # 训练数据
process_test_data(test_dir, "data/test_data.csv")  # 预测数据
