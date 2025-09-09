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
import shutil
from glob import glob
# ---------------- 新增引入 -----------------
# 请先安装: pip install scikit-image
from skimage.measure import label
# ------------------------------------------


# --- 损失函数定义 (保持不变) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', epsilon=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        probas = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = torch.pow((1 - probas), self.gamma)
        loss = self.alpha * focal_term * ce
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

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

# --- 全局参数 (保持不变) ---
TRAIN_START = 1
TRAIN_END = 500
LIMITATION = 1024
ORGANS = 15
CLASSES = 16

# --- 数据处理函数 (保持不变) ---
def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.relpath(os.path.join(root, file), directory))
    return file_paths

def read_nii_to_array(file_path):
    nii_targets = nib.load(file_path)
    return np.array(nii_targets.get_fdata())

# --- 模型定义 (保持不变) ---
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
        ch = [24, 48, 96, 192, 384]
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
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat((self.upconv4(b), e4), dim=1))
        d3 = self.dec3(torch.cat((self.upconv3(d4), e3), dim=1))
        d2 = self.dec2(torch.cat((self.upconv2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.upconv1(d2), e1), dim=1))
        return self.out_conv(d1)

# --- 数据集定义 (保持不变) ---
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

# ---------------- 核心修改: 将后处理函数加入 train.py -----------------
def postprocess_segmentation(segmentation_map):
    """
    对分割结果进行后处理，仅保留最大的连通组件。
    """
    if np.sum(segmentation_map) == 0:
        return segmentation_map
    labeled_mask = label(segmentation_map, connectivity=3)
    if labeled_mask.max() == 0:
        return segmentation_map
    largest_label = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
    cleaned_mask = np.zeros_like(segmentation_map, dtype=np.uint8)
    cleaned_mask[labeled_mask == largest_label] = 1
    return cleaned_mask

# ---------------- 核心修改: 评估函数加入 TTA 和后处理 -----------------
def get_prediction_for_eval(model, device, image):
    """
    用于周期性评估的推理函数，包含 TTA 和后处理
    """
    model.eval()
    image_normalized = np.clip(image, -LIMITATION, LIMITATION) / LIMITATION
    image_tensor = torch.tensor(image_normalized, dtype=torch.float32)
    inputs = image_tensor.permute(2, 0, 1).unsqueeze(1) # (Z, 1, X, Y)
    batchsize = 4
    num_slices = inputs.shape[0]
    
    all_outputs_probs = []
    with torch.no_grad():
        for flip in [False, True]:
            outputs_slices = []
            current_inputs = inputs.clone()
            if flip:
                current_inputs = torch.flip(current_inputs, [3])

            for idx in range(0, num_slices, batchsize):
                tail = min(idx + batchsize, num_slices)
                inputs_batch = current_inputs[idx:tail].to(device)
                outputs_batch = model(inputs_batch)
                
                if flip:
                    outputs_batch = torch.flip(outputs_batch, [3])
                
                outputs_slices.append(F.softmax(outputs_batch, dim=1).cpu())
            
            all_outputs_probs.append(torch.cat(outputs_slices, dim=0))

    avg_probs = torch.mean(torch.stack(all_outputs_probs), dim=0)
    final_pred = torch.argmax(avg_probs, dim=1)
    final_pred_numpy = final_pred.permute(1, 2, 0).numpy().astype(np.uint8)

    final_result_postprocessed = np.zeros_like(final_pred_numpy, dtype=np.int16)
    for organ_label in range(1, 16): 
        organ_mask = (final_pred_numpy == organ_label)
        if np.any(organ_mask):
            cleaned_organ_mask = postprocess_segmentation(organ_mask)
            final_result_postprocessed[cleaned_organ_mask == 1] = organ_label
            
    model.train() # 评估结束后，恢复训练模式
    return final_result_postprocessed


# --- 评分和评估主函数 (保持不变) ---
def compute_dice_metrics(predict, result, epsilon=1e-7):
    if predict.shape != result.shape: raise ValueError("预测标签和真实标签形状不相同")
    pixel = np.prod(result.shape[:-1])
    dice_scores = np.zeros(15)
    for k in range(15):
        pred_k, true_k = predict[..., k], result[..., k]
        intersection = np.sum(pred_k * true_k)
        sum_pred, sum_true = np.sum(pred_k), np.sum(true_k)
        if sum_true == 0 and sum_pred == 0:
            dice_scores[k] = 1.0
        else:
            dice_scores[k] = (2.0 * intersection) / (sum_pred + sum_true + epsilon)
    return dice_scores, pixel

def convert_to_onehot(label):
    if np.sum(label > 15) > 0: raise ValueError("标签中出现大于15的值")
    one_hot = np.zeros(label.shape + (15,), dtype=np.uint8)
    non_bg_indices = np.where(label > 0)
    one_hot[(*non_bg_indices, label[non_bg_indices] - 1)] = 1
    return one_hot

def grade_for_eval(result_path, predict_path):
    result_paths = sorted(glob(os.path.join(result_path, '*.nii.gz')))
    predict_paths = sorted(glob(os.path.join(predict_path, '*.nii.gz')))
    
    if not result_paths or not predict_paths or len(result_paths) != len(predict_paths):
        print("Warning: Label files or prediction files are missing or mismatched. Skipping grading.")
        return

    pixel_total, dice_scores_total = 0, np.zeros(15)
    for re_path, pr_path in tqdm(zip(result_paths, predict_paths), total=len(result_paths), desc="Grading"):
        re_label = read_nii_to_array(re_path).astype(np.uint8)
        pr_label = read_nii_to_array(pr_path).astype(np.uint8)
        re_onehot = convert_to_onehot(re_label)
        pr_onehot = convert_to_onehot(pr_label)
        dice_scores, pixel = compute_dice_metrics(pr_onehot, re_onehot)
        dice_scores_total += dice_scores * pixel
        pixel_total += pixel

    if pixel_total > 0:
        average_score = dice_scores_total / pixel_total
        total_score = 100 * np.mean(average_score)
        print("加权平均 Dice 分数:")
        formatted_score = [f"{s:.4f}" for s in average_score]
        print("[" + " ".join(formatted_score) + "]")
        print(f"总得分:\n{total_score:.4f}")
    else:
        print("Warning: Could not calculate score (pixel_total is zero).")

def run_evaluation(model, device, test_image_path, test_label_path, epoch_num):
    print(f"\n--- Running Evaluation for Epoch {epoch_num} (with TTA and Post-processing) ---")
    temp_predict_dir = "temp_predict_for_eval"
    if os.path.exists(temp_predict_dir): shutil.rmtree(temp_predict_dir)
    os.makedirs(temp_predict_dir)

    test_image_files = sorted(glob(os.path.join(test_image_path, "*.nii.gz")))
    if not test_image_files:
        print(f"Warning: No test images found in {test_image_path}. Skipping evaluation.")
        shutil.rmtree(temp_predict_dir)
        return

    for file_path in tqdm(test_image_files, desc=f"Predicting (Epoch {epoch_num})"):
        filename = os.path.basename(file_path)
        img = nib.load(file_path)
        # 调用我们新的、带后处理的预测函数
        pred_data = get_prediction_for_eval(model, device, np.array(img.get_fdata()))
        pred_img = nib.Nifti1Image(pred_data, img.affine, img.header)
        nib.save(pred_img, os.path.join(temp_predict_dir, filename))

    grade_for_eval(result_path=test_label_path, predict_path=temp_predict_dir)
    shutil.rmtree(temp_predict_dir)
    print("--- Evaluation Finished ---\n")


# --- 训练函数 (保持不变) ---
def model_train(dataloader, model_path, test_image_path, test_label_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded previous model weights.")

    criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_dice = DiceLoss()
    focal_weight, dice_weight = 0.6, 0.4
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs, batchsize = 15, 8
    
    batches_per_epoch = sum((len(images.squeeze(0)) + batchsize - 1) // batchsize for images, _ in dataloader)
    total_steps = batches_per_epoch * epochs
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=total_steps)
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="case")
        for images, targets in pbar:
            images, targets = images.squeeze(0).unsqueeze(1), targets.squeeze(0)
            for idx in range(0, len(images), batchsize):
                num_batches += 1
                batch_end = min(idx + batchsize, len(images))
                inputs_batch, targets_batch = images[idx:batch_end].to(device), targets[idx:batch_end].to(device)
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

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{epochs} finished, Average Loss: {avg_epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch + 1}")

        if (epoch + 1) % 2 == 0:
            if test_image_path and test_label_path:
                run_evaluation(model, device, test_image_path, test_label_path, epoch + 1)
            else:
                print("Warning: Test paths not provided, skipping intermediate evaluation.")
    
    torch.save(model.state_dict(), model_path)
    print("Final model saved.")


# --- 主函数 (保持不变) ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", dest="modelpath", type=str, help="Model path")
    parser.add_argument("-image", dest="imagepath", type=str, help="Images-Train path")
    parser.add_argument("-label", dest="labelpath", type=str, help="Labels-Train path")
    parser.add_argument("-test_image", dest="testimagepath", type=str, default="RIC22/imagesTs", help="Images-Test path")
    parser.add_argument("-test_label", dest="testlabelpath", type=str, default="RIC22/labelsTs", help="Labels-Test path")
    args = parser.parse_args()

    image_files = [f for f in get_all_file_paths(args.imagepath) if f.startswith("RIC_") and f.endswith(".nii.gz") and TRAIN_START <= int(f[4:8]) <= TRAIN_END]
    label_files = [f for f in get_all_file_paths(args.labelpath) if f.startswith("RIC_") and f.endswith(".nii.gz") and TRAIN_START <= int(f[4:8]) <= TRAIN_END]
    
    dataset = NiiDataset(image_files, label_files, args.imagepath, args.labelpath)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    
    model_train(dataloader, args.modelpath, args.testimagepath, args.testlabelpath)

if __name__ == "__main__":
    main()