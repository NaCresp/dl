"示例代码"

import os
import argparse
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #避免某些系统上的库冲突警告

# 训练集编号范围
TRAIN_START = 1
TRAIN_END = 500

# CT原始图像范围限制
LIMITATION = 1024  # -1024 ~ 1024
# 标签图像范围限制
ORGANS = 15  # 0 ~ 15
# 类别数量（15个器官 + 1个背景）
CLASSES = 16

def get_all_file_paths(directory):
    "遍历指定目录及其子目录，收集所有文件的相对路径"
    file_paths = []
    for root, _, files in os.walk(directory):  # os.walk遍历目录树
        for file in files:
            # 拼接完整路径并转换为相对路径
            file_paths.append(os.path.relpath(os.path.join(root, file), directory))
    return file_paths

def read_nii_to_array(file_path):
    "加载 .nii.gz 格式的医学图像文件, 提取数据并转为 numpy 数组"
    nii_targets = nib.load(file_path)
    return np.array(nii_targets.get_fdata())

# 定义UNet模型类
class UNet(nn.Module):
    """用于医学图像分割的UNet模型，包含3层编码器和3层解码器"""
    def __init__(self):
        super(UNet, self).__init__()

        # 编码器部分（下采样）
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 解码器部分（上采样，含跳跃连接）
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        # **核心修改**: 输出通道数改为CLASSES(16)，以匹配类别数
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, CLASSES, kernel_size=1)
        )

    def forward(self, x):
        "前向传播"
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d1 = self.dec1(e3)
        d1 = torch.cat([d1, e2], dim=1)

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e1], dim=1)

        output = self.dec3(d2)
        return output

# 定义自定义数据集类
class NiiDataset(Dataset):
    "处理医学图像数据集的自定义Dataset类，支持动态尺寸图像"
    def __init__(self, image_list, target_list, images_dir, labels_dir):
        self.image = image_list
        self.target = target_list
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.list = self.retain_common_items()
        self.len = len(self.list)

    def __len__(self):
        return self.len

    def retain_common_items(self):
        "筛选两个列表中同名的文件，确保图像与标签对应"
        image_set = set(self.image)
        target_set = set(self.target)
        common_items = image_set.intersection(target_set)
        return list(common_items)

    def __getitem__(self, idx):
        "加载并预处理单个数据样本"
        img_path = os.path.join(self.images_dir, self.list[idx])
        image = np.clip(read_nii_to_array(img_path), -LIMITATION, LIMITATION) / LIMITATION
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) # (D, H, W)

        label_path = os.path.join(self.labels_dir, self.list[idx])
        # **核心修改**: 加载标签为长整型，不进行归一化
        target = read_nii_to_array(label_path)
        target = np.clip(target, 0, ORGANS)
        target = torch.tensor(target, dtype=torch.long).permute(2, 0, 1) # (D, H, W)

        return image, target

def model_train(dataloader, model_path):
    "模型训练过程"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded previous model weights.")

    # **核心修改**: 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 21 # 增加训练轮次以更好地收敛
    batchsize = 4

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="case")
        for images, targets in pbar:
            images = images.squeeze(0).unsqueeze(1) # (D, 1, H, W)
            targets = targets.squeeze(0) # (D, H, W)

            for idx in range(0, len(images), batchsize):
                batch_end = min(idx + batchsize, len(images))
                inputs_batch = images[idx:batch_end].to(device)
                targets_batch = targets[idx:batch_end].to(device) # (B, H, W)

                optimizer.zero_grad()
                outputs = model(inputs_batch) # (B, 16, H, W)
                loss = criterion(outputs, targets_batch)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} finished, Average Loss: {avg_epoch_loss:.4f}")

        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch + 1}")
    
    torch.save(model.state_dict(), model_path)
    print("Final model saved.")


def main():
    "创建数据集, 训练并保存模型"
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", dest="modelpath", type=str, help="Model path")
    parser.add_argument("-image", dest="imagepath", type=str, help="Images-Train path")
    parser.add_argument("-label", dest="labelpath", type=str, help="Labels-Train path")
    args = parser.parse_args()

    images_dir = args.imagepath
    labels_dir = args.labelpath

    image_files = [f for f in get_all_file_paths(images_dir)
                   if (f.startswith("RIC_") and f.endswith(".nii.gz")
                       and int(f[4:8]) >= TRAIN_START and int(f[4:8]) <= TRAIN_END)]
    label_files = [f for f in get_all_file_paths(labels_dir)
                   if (f.startswith("RIC_") and f.endswith(".nii.gz")
                       and int(f[4:8]) >= TRAIN_START and int(f[4:8]) <= TRAIN_END)]

    dataset = NiiDataset(image_files, label_files, images_dir, labels_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    model_train(dataloader, args.modelpath)

if __name__ == "__main__":
    main()