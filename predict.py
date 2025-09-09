"""
调用训练好的模型生成预测结果 (应用TTA和后处理)
"""

import os
import argparse
from glob import glob
import torch
from train import UNet 
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F
# ---------------- 新增引入 -----------------
# 请先安装: pip install scikit-image
from skimage.measure import label

# ------------------------------------------

# 处理命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-model", dest="modelpath", type=str, help="Model path")
parser.add_argument("-test", dest="tspath", type=str, help="Images-Test path")
parser.add_argument("-predict", dest="prpath", type=str, help="Predict path")
args = parser.parse_args()

MODEL_PATH = args.modelpath
TEST_PATH = args.tspath
PREDICT_PATH = args.prpath

# CT原始图像范围限制
LIMITATION = 1024  # -1024 ~ 1024

# 载入模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() #设置为评估模式


# ---------------- 核心修改: 新增后处理函数 -----------------
def postprocess_segmentation(segmentation_map):
    """
    对分割结果进行后处理，仅保留最大的连通组件。
    这能有效去除小的、孤立的噪声点。
    """
    if np.sum(segmentation_map) == 0:
        return segmentation_map # 如果没有分割区域，直接返回

    # 标记连通组件
    labeled_mask = label(segmentation_map, connectivity=3)
    
    # 如果没有找到组件，也直接返回
    if labeled_mask.max() == 0:
        return segmentation_map

    # 计算每个组件的大小，找到最大的那个
    largest_label = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
    
    # 创建一个新的掩码，只保留最大的组件
    cleaned_mask = np.zeros_like(segmentation_map, dtype=np.uint8)
    cleaned_mask[labeled_mask == largest_label] = 1
            
    return cleaned_mask


# ---------------- 核心修改: 修改推理函数以支持 TTA 和后处理 -----------------
def get_result(image):
    """
    调用神经网络进行推理, 生成预测结果
    - 应用测试时增强 (TTA): 水平翻转
    - 应用后处理: 保留最大连通域
    """
    image_normalized = np.clip(image, -LIMITATION, LIMITATION) / LIMITATION
    image_tensor = torch.tensor(image_normalized, dtype=torch.float32)
    inputs = image_tensor.permute(2, 0, 1).unsqueeze(1)  # (Z, 1, X, Y)
    
    batchsize = 4
    num_slices = inputs.shape[0]
    
    all_outputs_probs = []
    with torch.no_grad():
        # TTA策略: 原始图像 + 水平翻转图像
        for flip in [False, True]:
            outputs_slices = []
            current_inputs = inputs.clone()
            if flip:
                # 沿宽度维度(dim=3)进行翻转
                current_inputs = torch.flip(current_inputs, [3])

            for idx in range(0, num_slices, batchsize):
                tail = min(idx + batchsize, num_slices)
                inputs_batch = current_inputs[idx:tail].to(device)
                outputs_batch = model(inputs_batch)
                
                if flip:
                    # 将翻转后的结果再次翻转回来，以对齐原始图像
                    outputs_batch = torch.flip(outputs_batch, [3])
                
                # 将logits转换为概率，并移动到CPU
                outputs_slices.append(F.softmax(outputs_batch, dim=1).cpu())
            
            all_outputs_probs.append(torch.cat(outputs_slices, dim=0))

    # 对TTA的多次预测结果（概率）进行平均
    avg_probs = torch.mean(torch.stack(all_outputs_probs), dim=0) # (Z, 16, X, Y)
    
    # 根据平均后的概率，获取最终的类别预测
    final_pred = torch.argmax(avg_probs, dim=1) # (Z, X, Y)
    final_pred_numpy = final_pred.permute(1, 2, 0).numpy().astype(np.uint8) # (X, Y, Z)

    # --- 后处理步骤 ---
    final_result_postprocessed = np.zeros_like(final_pred_numpy, dtype=np.int16)
    # 对背景(0)之外的每个器官类别(1-15)分别进行后处理
    for organ_label in range(1, 16): 
        organ_mask = (final_pred_numpy == organ_label)
        if np.any(organ_mask):
            cleaned_organ_mask = postprocess_segmentation(organ_mask)
            final_result_postprocessed[cleaned_organ_mask == 1] = organ_label
            
    return final_result_postprocessed


# --- 主程序循环 (保持不变) ---
# 建立预测文件夹, 并读取所有测试文件
os.makedirs(PREDICT_PATH, exist_ok=True)
file_list = sorted(glob(os.path.join(TEST_PATH, "*.nii.gz")))

for file_path in tqdm(file_list, desc="Predicting"):
    filename = os.path.basename(file_path)
    # 读取图像
    img = nib.load(file_path)
    img_data = np.array(img.get_fdata())
    affine = img.affine
    header = img.header
    # 推理
    pred = get_result(img_data)
    # 保存预测结果
    pred_img = nib.Nifti1Image(pred, affine, header)
    nib.save(pred_img, os.path.join(PREDICT_PATH, filename))