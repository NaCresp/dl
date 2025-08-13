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
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 卷积层：输入通道1，输出64
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 二次卷积
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),  # 最大池化，尺寸减半
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 通道数翻倍
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),  # 再次池化
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 通道数继续增加
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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 双线性上采样恢复尺寸
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),  # 注意输入通道为256（128+128跳跃连接）
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, padding=1),   # 输入通道128（64+64跳跃连接）
            nn.Conv2d(1, 1, kernel_size=1)  # 1x1卷积调整输出通道
        )

    def forward(self, x):
        "前向传播"
        # 编码过程
        e1 = self.enc1(x)   # 初始特征图 (B, 64, H, W)
        e2 = self.enc2(e1)  # 下采样后 (B, 128, H/2, W/2)
        e3 = self.enc3(e2)  # 进一步下采样 (B, 256, H/4, W/4)

        # 解码过程（结合跳跃连接）
        d1 = self.dec1(e3)  # 上采样到H/2, W/2
        d1 = torch.cat([d1, e2], dim=1)  # 与编码器第二层输出拼接（通道维度）

        d2 = self.dec2(d1)  # 上采样到H, W
        d2 = torch.cat([d2, e1], dim=1)  # 与编码器第一层输出拼接

        output = self.dec3(d2)  # 最终输出 (B, 1, H, W)
        return output

# 定义自定义数据集类
class NiiDataset(Dataset):
    "处理医学图像数据集的自定义Dataset类，支持动态尺寸图像"
    def __init__(self, image_list, target_list, images_dir, labels_dir):
        self.image = image_list
        self.target = target_list
        self.images_dir = images_dir  # 图像文件目录
        self.labels_dir = labels_dir   # 标签文件目录
        self.list = self.retain_common_items()  # 确保图像和标签文件名一致
        self.len = len(self.list)

    def __len__(self):
        return self.len  # 返回数据集大小

    def retain_common_items(self):
        "筛选两个列表中同名的文件，确保图像与标签对应"
        image_set = set(self.image)
        target_set = set(self.target)
        common_items = image_set.intersection(target_set)  # 求交集
        return list(common_items)

    def __getitem__(self, idx):
        "加载并预处理单个数据样本"
        # 加载图像
        img_path = os.path.join(self.images_dir, self.list[idx])
        image = np.clip(read_nii_to_array(img_path), -LIMITATION, LIMITATION) / LIMITATION # 归一化
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # 添加通道维度(C=1)

        # 加载标签
        label_path = os.path.join(self.labels_dir, self.list[idx])
        target = 2 * np.clip(read_nii_to_array(label_path), 0, ORGANS) / ORGANS - 1
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

        return image, target  # 返回图像-标签对

def model_train(dataloader, model_path):
    "模型训练过程"
    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    # 读取之前参数
    if os.path.exists(model_path):
        # 加载包含模型参数、优化器状态的完整检查点
        model.load_state_dict(torch.load(model_path))
    criterion = nn.MSELoss()               # 使用均方误差损失（适用于回归任务）
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam优化器

    epochs = 4 * 5 + 1   # 训练轮次
    batchsize = 4        # 实际批大小（通过切片实现）

    for epoch in range(epochs):
        model.train()
        # 使用进度条遍历数据
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            # 调整数据维度：(B, C, H, W, D) -> (D, B, C, H, W)（假设处理三维切片）
            inputs = inputs.squeeze().transpose(2, 0).unsqueeze(1)  # 去除冗余维度并转置
            targets = targets.squeeze().transpose(2, 0).unsqueeze(1)

            # 分批处理（可能因内存限制）
            for idx in range(0, len(inputs), batchsize):
                # 截取当前小批次
                batch_end = min(idx + batchsize, len(inputs))
                inputs_batch = inputs[idx:batch_end]
                targets_batch = targets[idx:batch_end]

                # 迁移数据到设备并前向传播
                inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)
                optimizer.zero_grad()  # 清零梯度
                outputs = model(inputs_batch)
                loss = criterion(outputs, targets_batch)

                # 反向传播与优化
                loss.backward()
                optimizer.step()

            # 打印当前损失
            tqdm.write(f"Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        # 每5轮保存模型
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch + 1}")

def main():
    "创建数据集, 训练并保存模型"
    # 处理命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", dest="modelpath", type=str, help="Model path")
    parser.add_argument("-image", dest="imagepath", type=str, help="Images-Train path")
    parser.add_argument("-label", dest="labelpath", type=str, help="Labels-Train path")
    args = parser.parse_args()

    images_dir = args.imagepath
    labels_dir = args.labelpath

    # 过滤训练集文件: 文件类型为".nii.gz"且编号在选定区间内
    image_files = [f for f in get_all_file_paths(images_dir)
                   if (f.startswith("RIC_") and f.endswith(".nii.gz")
                       and int(f[4:8]) >= TRAIN_START and int(f[4:8]) <= TRAIN_END)]
    label_files = [f for f in get_all_file_paths(labels_dir)
                   if (f.startswith("RIC_") and f.endswith(".nii.gz")
                       and int(f[4:8]) >= TRAIN_START and int(f[4:8]) <= TRAIN_END)]

    # 创建数据集和数据加载器
    dataset = NiiDataset(image_files, label_files, images_dir, labels_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # 批处理设为1（可能因尺寸不固定）

    # 训练模型
    model_train(dataloader, args.modelpath)

if __name__ == "__main__":
    main()
