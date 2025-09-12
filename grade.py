"评分代码"

import os
import argparse
from glob import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm

def compute_dice_metrics(predict, result, epsilon=1e-7):
    """
    计算预测结果与真实标签间的 Dice 指标和总像素数
    
    Args
    ----
    predict : (X, Y, Z, 15) ndarray
        预测结果的 one-hot 编码
    result : (X, Y, Z, 15) ndarray
        真实结果的 one-hot 编码
    epsilon : float
        平滑系数, 默认为 1e-7

    Returns
    -------
    dice_scores : (15, ) ndarray
        每个器官的 Dice 分数
    pixel : int
        图像空间像素数
    """
    # 验证输入形状
    if predict.shape != result.shape:
        raise ValueError("预测标签和真实标签形状不相同")

    # 计算图像空间像素数
    pixel = np.prod(result.shape[:-1])

    # 初始化DICE分数数组
    dice_scores = np.zeros(15)

    # 遍历每个器官通道
    for k in range(15):
        pred_k = predict[..., k]
        true_k = result[..., k]

        # 计算交集和各自的和
        intersection = np.sum(pred_k * true_k)
        sum_pred = np.sum(pred_k)
        sum_true = np.sum(true_k)

        if sum_true == 0 and sum_pred == 0: # 处理特殊情况
            dice_scores[k] = 1.0
        else:
            dice_scores[k] = (2.0 * intersection) / (sum_pred + sum_true + epsilon)

    return dice_scores, pixel

def read_nii_to_array(file_path):
    "加载 .nii.gz 格式的医学图像文件, 提取数据并转为 numpy 数组"
    nii_targets = nib.load(file_path)
    return np.array(nii_targets.get_fdata()).astype(np.uint8)

def convert_to_onehot(label):
    """
    把三维标签图像转化为 one-hot 编码
    
    Args
    ----
    label : (X, Y, Z) ndarray
        三维整数标签图像, 值范围为 0-15

    Returns
    -------
    one_hot : (X, Y, Z, 15) ndarray
    """
    # 越界检测
    if np.sum(label > 15) > 0:
        raise ValueError("标签中出现大于15的值")

    # 创建空数组
    one_hot = np.zeros(label.shape + (15,), dtype=np.uint8)
    # 一次性设置所有非背景像素
    one_hot[(*np.where(label > 0), label[label > 0] - 1)] = 1

    return one_hot

def load_label_dataset(result_path, predict_path):
    """
    读取标签文件, 并转化为 one-hot 编码
    
    Args
    ----
    result_path, predict_path : 
        真实标签与预测标签的文件夹路径

    Returns
    -------
    result, predict : List
        真实标签与预测标签的数据列表, 每个元素分别为 (X, Y, Z, 15) ndarray
    """
    # 验证文件夹存在性
    if not os.path.isdir(result_path):
        raise FileNotFoundError(f"真实标签文件夹不存在: {result_path}")
    if not os.path.isdir(predict_path):
        raise FileNotFoundError(f"预测标签文件夹不存在: {predict_path}")

    # 获取所有标签文件路径
    result_paths = sorted(glob(os.path.join(result_path, '*.nii.gz')))
    predict_paths = sorted(glob(os.path.join(predict_path, '*.nii.gz')))
    if not result_paths:
        raise FileNotFoundError(f"真实标签文件夹中没有找到图像文件: {result_path}")
    if not predict_paths:
        raise FileNotFoundError(f"预测标签文件夹中没有找到标签文件: {predict_path}")

    # 验证标签文件匹配
    result_files = [os.path.basename(p) for p in result_paths]
    predict_files = [os.path.basename(p) for p in predict_paths]
    if result_files != predict_files:
        missing_in_predict = set(result_files) - set(predict_files)
        missing_in_result = set(predict_files) - set(result_files)
        error_message = (
            "两个文件夹中的 .nii.gz 文件名不完全相同.\n"
            f"在 {predict_path} 中缺失的文件: {missing_in_predict}\n"
            f"在 {result_path} 中缺失的文件: {missing_in_result}"
        )
        raise ValueError(error_message)

    # 逐个加载文件
    result, predict = [], []
    for re_path, pr_path in tqdm(zip(result_paths, predict_paths), total=len(result_paths),
                                 desc="Loading labels"):
        result.append(convert_to_onehot(read_nii_to_array(re_path)))
        predict.append(convert_to_onehot(read_nii_to_array(pr_path)))

    return result, predict

def grade(result_path, predict_path):
    "评分"
    # 加载数据集
    result, predict = load_label_dataset(result_path, predict_path)

    # 计算DICE指标并用像素进行加权平均
    pixel_total = 0
    dice_scores_total = np.zeros(15)
    for re, pr in tqdm(zip(result, predict), total=len(result),
                       desc="Grading"):
        dice_scores, pixel = compute_dice_metrics(pr, re)
        dice_scores_total += dice_scores * pixel
        pixel_total += pixel

    # 计算加权平均 Dice 分数
    average_score = dice_scores_total / pixel_total if pixel_total > 0 else np.zeros(15)
    print("加权平均 Dice 分数:")
    formatted_score = [f"{s:.4f}" for s in average_score]
    print("[" + " ".join(formatted_score) + "]")

    # 计算总得分
    total_score = 100 * np.mean(average_score)
    print(f"总得分:\n{total_score:.4f}")

if __name__ == "__main__":
    # 处理命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-result", dest="repath", type=str, help="Lables-Test path")
    parser.add_argument("-predict", dest="prpath", type=str, help="Predict path")
    args = parser.parse_args()

    grade(args.repath, args.prpath)
