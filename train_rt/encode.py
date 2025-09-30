import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml


# 假设以下模块与训练代码在同一目录
from dataset_RT_GCD import RTDataset
from model import VQRTCodec

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_yaml_config(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def encode_rt(config, input_rt):
    """
    对指定的RT数据进行编码和解码

    Args:
        config: 配置文件
        input_rt: 输入的RT数据，形状为(7,)的numpy数组，前3个为平移，后4个为四元数

    Returns:
        indices: 编码后的索引
        decoded_rt: 解码后的RT数据
    """
    # 初始化模型
    model = VQRTCodec(config).to(device)
    model.load_state_dict(torch.load(config["test"]["test_model_path"]))
    model.eval()

    # 将输入转换为tensor并添加batch维度
    input_tensor = torch.tensor(input_rt, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        # 编码
        z = model.encoder(input_tensor)
        quantized, indices, _ = model.codebook(z)
        # 解码
        decoded = model.decoder(quantized)

        # 转换为numpy数组
        indices_np = [idx.cpu().numpy() for idx in indices]
        decoded_np = decoded.squeeze(0).cpu().numpy()

    return indices_np, decoded_np


def quaternion_to_euler(q, degrees=True):
    """
    将四元数转换为欧拉角（roll, pitch, yaw）
    输入形状为 (4,) 的numpy数组
    """
    w, x, y, z = q

    # X轴旋转（roll）
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Y轴旋转（pitch）
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)

    # Z轴旋转（yaw）
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if degrees:
        return np.degrees([roll, pitch, yaw])
    return np.array([roll, pitch, yaw])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode and decode RT data")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config_rt.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--rt",
        type=float,
        nargs=7,
        help="RT values: 3 translation + 4 quaternion values",
    )
    args = parser.parse_args()

    config = read_yaml_config(args.config_path)

    # 如果命令行没有提供RT值，使用默认值
    if args.rt is None:
        # 默认RT值: [tx, ty, tz, qw, qx, qy, qz]
        input_rt = np.array([-40.08815, 40.77250000000001, -25.0, 1.0, 0.0, 0.0, 0.0])
    else:
        input_rt = np.array(args.rt)

    # 编码和解码
    indices, decoded_rt = encode_rt(config, input_rt)

    # 计算欧拉角
    input_euler = quaternion_to_euler(input_rt[3:])
    decoded_euler = quaternion_to_euler(decoded_rt[3:])

    # 输出结果
    print("\n===== 输入RT数据 =====")
    print(f"平移 (tx, ty, tz): {input_rt[:3]}")
    print(f"四元数 (qw, qx, qy, qz): {input_rt[3:]}")
    print(f"欧拉角 (roll, pitch, yaw): {input_euler}")

    print("\n===== 编码索引 =====")
    for i, idx in enumerate(indices):
        print(f"量化器 {i}: {idx}")

    print("\n===== 解码RT数据 =====")
    print(f"平移 (tx, ty, tz): {decoded_rt[:3]}")
    print(f"四元数 (qw, qx, qy, qz): {decoded_rt[3:]}")
    print(f"欧拉角 (roll, pitch, yaw): {decoded_euler}")

    # 计算误差
    trans_error = np.linalg.norm(input_rt[:3] - decoded_rt[:3])
    quat_error = np.linalg.norm(input_rt[3:] - decoded_rt[3:])
    euler_error = np.linalg.norm(input_euler - decoded_euler)

    print("\n===== 误差 =====")
    print(f"平移误差: {trans_error:.6f}")
    print(f"四元数误差: {quat_error:.6f}")
    print(f"欧拉角误差: {euler_error:.6f}")
