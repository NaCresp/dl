
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
from torch.cuda.amp import GradScaler, autocast

# --- 损失函数定义 (最终版: Focal + Dice) ---

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', epsilon=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        # 计算基础的交叉熵损失，但不进行聚合
        ce = self.ce_loss(logits, targets)
        # 计算概率
        probas = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        # 计算Focal Loss的调制因子
        focal_term = torch.pow((1 - probas), self.gamma)
        # 最终的损失
        loss = self.alpha * focal_term * ce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probas = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            probas = probas[:, 1:, :, :]
            targets_one_hot = targets_one_hot[:, 1:, :, :]
        
        intersection = torch.sum(probas * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probas + targets_one_hot, dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return (1 - dice_score).mean()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- 全局参数 ---
TRAIN_START = 1
TRAIN_END = 500
LIMITATION = 1024
ORGANS = 15
CLASSES = 16

# --- 数据处理函数 ---
def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.relpath(os.path.join(root, file), directory))
    return file_paths

def read_nii_to_array(file_path):
    nii_targets = nib.load(file_path)
    return np.array(nii_targets.get_fdata())

# --- 模型定义 ---
class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        if in_channels == out_channels:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv_block(x)
        out += residual
        return self.relu(out)

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=16):
        super(UNet, self).__init__()
        # 使用V3版本的增强模型
        ch = [48, 96, 192, 384, 768]
        self.enc1 = ResConvBlock(in_channels, ch[0])
        self.enc2 = ResConvBlock(ch[0], ch[1])
        self.enc3 = ResConvBlock(ch[1], ch[2])
        self.enc4 = ResConvBlock(ch[2], ch[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResConvBlock(ch[3], ch[4])
        self.upconv4 = nn.ConvTranspose2d(ch[4], ch[3], kernel_size=2, stride=2)
        self.dec4 = ResConvBlock(ch[4], ch[3])
        self.upconv3 = nn.ConvTranspose2d(ch[3], ch[2], kernel_size=2, stride=2)
        self.dec3 = ResConvBlock(ch[3], ch[2])
        self.upconv2 = nn.ConvTranspose2d(ch[2], ch[1], kernel_size=2, stride=2)
        self.dec2 = ResConvBlock(ch[2], ch[1])
        self.upconv1 = nn.ConvTranspose2d(ch[1], ch[0], kernel_size=2, stride=2)
        self.dec1 = ResConvBlock(ch[1], ch[0])
        self.out_conv = nn.Conv2d(ch[0], n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
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
        return self.out_conv(d1)

# --- 数据集定义 ---
class NiiDataset(Dataset):
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
        return list(set(self.image).intersection(set(self.target)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.list[idx])
        image = np.clip(read_nii_to_array(img_path), -LIMITATION, LIMITATION) / LIMITATION
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        label_path = os.path.join(self.labels_dir, self.list[idx])
        target = np.clip(read_nii_to_array(label_path), 0, ORGANS)
        target = torch.tensor(target, dtype=torch.long).permute(2, 0, 1)
        return image, target

# --- 训练函数 ---
def model_train(dataloader, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded previous model weights.")

    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_dice = DiceLoss()
    focal_weight = 0.6 # 提高Focal的权重，强制关注困难样本
    dice_weight = 0.4

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    epochs = 40 # 增加训练轮数，给Focal Loss更多时间发挥作用
    batchsize = 8

    total_steps = 0
    print("Calculating total steps for the scheduler...")
    for images, _ in tqdm(dataloader, desc="Pre-calculating steps"):
        num_batches_in_case = (len(images.squeeze(0)) + batchsize - 1) // batchsize
        total_steps += num_batches_in_case
    total_steps *= epochs
    print(f"Total training steps calculated: {total_steps}")
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=total_steps)
    scaler = GradScaler()

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
                optimizer.zero_grad(set_to_none=True)
                
                with autocast():
                    outputs = model(inputs_batch)
                    loss_focal = criterion_focal(outputs, targets_batch)
                    loss_dice = criterion_dice(outputs, targets_batch)
                    loss = focal_weight * loss_focal + dice_weight * loss_dice

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", focal=f"{loss_focal.item():.4f}", dice=f"{loss_dice.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")
        
        avg_epoch_loss = epoch_loss / len(pbar)
        print(f"Epoch {epoch + 1}/{epochs} finished, Average Loss: {avg_epoch_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch + 1}")
    
    torch.save(model.state_dict(), model_path)
    print("Final model saved.")

# --- 主函数 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", dest="modelpath", type=str, help="Model path")
    parser.add_argument("-image", dest="imagepath", type=str, help="Images-Train path")
    parser.add_argument("-label", dest="labelpath", type=str, help="Labels-Train path")
    args = parser.parse_args()

    image_files = [f for f in get_all_file_paths(args.imagepath) if f.startswith("RIC_") and f.endswith(".nii.gz") and TRAIN_START <= int(f[4:8]) <= TRAIN_END]
    label_files = [f for f in get_all_file_paths(args.labelpath) if f.startswith("RIC_") and f.endswith(".nii.gz") and TRAIN_START <= int(f[4:8]) <= TRAIN_END]
    dataset = NiiDataset(image_files, label_files, args.imagepath, args.labelpath)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    model_train(dataloader, args.modelpath)

if __name__ == "__main__":
    main()