"""
调用训练好的模型生成预测结果
"""

import os
import argparse
from glob import glob
import torch
from train import UNet
import numpy as np
import nibabel as nib
from tqdm import tqdm

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
# 标签图像范围限制
ORGANS = 15  # 0 ~ 15

# 载入模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval() #设置为评估模式

def get_result(image):
    """
    调用神经网络进行推理, 生成预测结果

    Args
    ----
    image : (X, Y, Z) ndarray
        原始图像, 格式与 images 对应文件夹中图像相同

    Returns
    -------
    result : (X, Y, Z) ndarray
        分割后图像, 格式与 labels 对应文件夹中图像相同
    """
    # 将 image 转换为 torch tensor
    image = np.clip(image, -LIMITATION, LIMITATION) / LIMITATION # 归一化
    image = torch.tensor(image, dtype=torch.float32, device=device)  # (X, Y, Z)
    inputs = image.permute(2, 0, 1).unsqueeze(1)  # (Z, 1, X, Y)

    # 分批处理
    batchsize = 4
    num_slices = inputs.shape[0]
    result = torch.zeros_like(inputs, device='cpu')  # (Z, 1, X, Y)
    with torch.no_grad():
        for idx in range(0, num_slices, batchsize):
            tail = min(idx + batchsize, len(inputs))
            inputs_batch = inputs[idx:tail]  # (B, 1, X, Y)
            outputs_batch = model(inputs_batch.to(device)) #推理
            result[idx:tail] = outputs_batch.cpu() #拼接结果

    # 去掉多余的维度, 并转化为整数 numpy 数组
    result = result.squeeze(1).numpy()  # (Z, X, Y)
    result = np.transpose(result, (1, 2, 0))  # (X, Y, Z)
    result = np.clip(np.round((result + 1) * ORGANS / 2), 0, ORGANS).astype(np.int32) #去归一化并四舍五入

    return result

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
