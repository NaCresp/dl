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
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # 将logits通过softmax转换成概率
        probas = F.softmax(logits, dim=1)
        
        # 将整数标签 targets 转换为 one-hot 编码
        # targets 的形状是 (B, H, W)，需要转换为 (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=CLASSES).permute(0, 3, 1, 2).float()

        dice_loss = 0.0
        # 遍历每个类别（不包括背景）
        for i in range(1, CLASSES):
            probas_i = probas[:, i, :, :]
            targets_i = targets_one_hot[:, i, :, :]
            
            intersection = torch.sum(probas_i * targets_i)
            union = torch.sum(probas_i) + torch.sum(targets_i)
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice)
            
        # 返回所有器官Dice Loss的平均值
        return dice_loss / (CLASSES - 1)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', epsilon=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # 将logits通过softmax转换成概率
        probas = F.softmax(logits, dim=1)
        # 获取对应target类别的概率
        probas = probas.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 计算focal loss
        log_probas = torch.log(probas + self.epsilon)
        loss = -self.alpha * torch.pow((1 - probas), self.gamma) * log_probas
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

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

# 定义一个卷积块，方便复用
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), # 添加BatchNorm，加速收敛，提升稳定性
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# 定义UNet模型类
class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=16):
        super(UNet, self).__init__()

        # 编码器
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)

        # 中间瓶颈层
        self.bottleneck = conv_block(512, 1024)

        # 解码器
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 瓶颈
        b = self.bottleneck(self.pool(e4))

        # 解码与跳跃连接
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        output = self.out_conv(d1)
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

    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_dice = DiceLoss()
    # 可以给Focal和Dice分配不同的权重
    focal_weight = 0.5
    dice_weight = 0.5

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)


    epochs = 21
    batchsize = 4

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="case")
        for images, targets in pbar:
            images = images.squeeze(0).unsqueeze(1)
            targets = targets.squeeze(0)

            for idx in range(0, len(images), batchsize):
                batch_end = min(idx + batchsize, len(images))
                inputs_batch = images[idx:batch_end].to(device)
                targets_batch = targets[idx:batch_end].to(device)

                optimizer.zero_grad()
                outputs = model(inputs_batch)
                
                loss_focal = criterion_focal(outputs, targets_batch)
                loss_dice = criterion_dice(outputs, targets_batch)
                loss = focal_weight * loss_focal + dice_weight * loss_dice

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", focal=f"{loss_focal.item():.4f}", dice=f"{loss_dice.item():.4f}")
        
        scheduler.step()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} finished, Average Loss: {avg_epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch+1) % 10 == 0: # 调整保存频率
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
    