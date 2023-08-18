import os

import pandas as pd

import cv2
import numpy as np

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import (
    Embedding,
    LSTM,
    Input,
    Flatten,
    Dense,
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

from PIL import Image
from progressbar import ProgressBar
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

pbar = ProgressBar()

tags_len = 30
pos_images = []
neg_images = []
pos_labels = []
neg_labels = []
y = []
features = []
light_dict = {}
pos_dir = ""
neg_dir = ""
# 读取图像和标签数据
pos_dir = "C:/Users/younger/Pictures/ApplePic/photo_tag/useful"
neg_dir = "C:/Users/younger/Pictures/ApplePic/photo_tag/useless"

kernel1 = ([-1, -1, -1], [-1, 8, -1], [-1, -1, -1])


kernel_conv = np.array(kernel1)

print("正在获取并预处理照片...\n")
print("positive images:\n")

for f in tqdm(os.listdir(pos_dir)):
    if f.endswith(".txt"):
        with open(os.path.join(pos_dir, f)) as file:
            lines = file.read().splitlines()
            times=0
            for line in lines:
                times+=1
                line = line.split(",")
                pos_labels.append(line)
                if times>tags_len:
                    break
    elif f.endswith(".png"):
        img_lines = cv2.imread(os.path.join(pos_dir, f))
        # conv_img = cv2.filter2D(img_lines, cv2.CV_64F, kernel_conv)
        pos_images.append(img_lines)

print("negative images:\n")
for f in tqdm(os.listdir(neg_dir)):
    if f.endswith(".txt"):
        with open(os.path.join(neg_dir, f)) as file:
            lines = file.read().splitlines()
            for line in lines:
                line = line.split(",")
                neg_labels.append(lines)
    if f.endswith(".png"):
        img_lines = cv2.imread(os.path.join(neg_dir, f))
        neg_images.append(img_lines)


def extract_image_features(image):
    # 加载 VGG16 模型
    model = VGG16(weights="imagenet", include_top=False)
    # 预处理图像
    img = cv2.resize(image, (512, 512))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # print(img)
    # 提取特征向量
    # print("正在提取图像特征向量...\n")
    img_features = model.predict(img, verbose=0)
    img_features = img_features.flatten()
    return img_features


# 提取标签特征向量
def extract_tag_features(labels):
    # print("正在提取标签特征向量...\n")
    # 构建标签向量
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(labels)
    sequences = tokenizer.texts_to_sequences(labels)
    tag_features = pad_sequences(sequences, maxlen=tags_len)
    # 构建 LSTM 模型
    model = Sequential()
    model.add(
        Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=64,
            input_length=tags_len,
        )
    )
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    tag_features = model.predict(tag_features, verbose=0)
    target_shape = (tags_len, 64)
    if tag_features.shape != target_shape:
        new_tag_features = np.zeros(target_shape)
        new_tag_features[
            : tag_features.shape[0], : tag_features.shape[1]
        ] = tag_features
        tag_features = new_tag_features  # print(tag_features.shape)
    return tag_features


scaler = StandardScaler()
print("正在提取特征向量...\n")
y = []
features = []
for img, lables in tqdm(zip(pos_images, pos_labels)):
    img_features = extract_image_features(img)

    tag_features = extract_tag_features(lables)
    signle_features = np.concatenate(
        (tag_features, np.tile(img_features, (len(tag_features), 1))), axis=1
    )

    # 标准化特征矩阵
    signle_features = scaler.fit_transform(signle_features)
    y.append(1)
    features.append(signle_features)

for img, lables in tqdm(zip(neg_images, neg_labels)):
    img_features = extract_image_features(img)
    tag_features = extract_tag_features(lables)
    signle_features = np.concatenate(
        (tag_features, np.tile(img_features, (len(tag_features), 1))), axis=1
    )

    # 标准化特征矩阵
    signle_features = scaler.fit_transform(signle_features)
    y.append(0)
    features.append(signle_features)
print("特征提取完毕！\n")


features_1d = [feature.flatten() for feature in features]


# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42
)

# 训练模型
accuracy = pd.DataFrame(columns=["accuracy"])
precision = pd.DataFrame(columns=["precision"])
recall = pd.DataFrame(columns=["recall"])
f1 = pd.DataFrame(columns=["f1"])
kernel_list = ["poly", "rbf", "sigmoid"]
C_list = [0.1, 1, 10, 100]
print("kernel\tC\taccuracy\tprecision\trecall\tf1")
for kernel in tqdm(kernel_list):
    for C in tqdm(C_list):
        print("正在训练模型...\n")
        print("kernel=", kernel, "C=", C)
        clf = SVC(kernel=kernel, C=C)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy.append(
            {"accuracy": accuracy_score(y_test, y_pred)}, ignore_index=True
        )
        precision = precision.append(
            {"precision": precision_score(y_test, y_pred)}, ignore_index=True
        )
        recall = recall.append(
            {"recall": recall_score(y_test, y_pred)}, ignore_index=True
        )
        f1 = f1.append({"f1": f1_score(y_test, y_pred)}, ignore_index=True)

# 保存表格
from itertools import product

df = pd.concat([accuracy, precision, recall, f1], axis=1)
df.columns = ["accuracy", "precision", "recall", "f1"]
df.index = [(kernel, C) for kernel, C in product(kernel_list, C_list)]

# 打印表格
print(df)
