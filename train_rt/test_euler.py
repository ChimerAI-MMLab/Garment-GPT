import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml

from dataset_RT_GCD_euler import RTDataset
from model import VQRTCodec

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_yaml_config(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def visualize_sample(true_rt, pred_rt, save_path):
    """可视化单个样本的预测结果（欧拉角输出）"""
    plt.figure(figsize=(12, 6))

    # 平移部分可视化
    plt.subplot(1, 2, 1)
    labels = ["X", "Y", "Z"]
    x = np.arange(3)
    plt.bar(x - 0.1, true_rt[:3], width=0.2, label="True")
    plt.bar(x + 0.1, pred_rt[:3], width=0.2, label="Pred")
    plt.xticks(x, labels)
    plt.title("Translation Comparison")
    plt.legend()

    # 欧拉角部分可视化
    plt.subplot(1, 2, 2)
    labels = ["Roll", "Pitch", "Yaw"]
    x = np.arange(3)
    # 保证角度在0-360范围
    true_euler = np.where(true_rt[3:] < 0, true_rt[3:] + 360, true_rt[3:])
    pred_euler = np.where(pred_rt[3:] < -1, pred_rt[3:] + 360, pred_rt[3:])
    plt.bar(x - 0.1, true_euler, width=0.2, label="True")
    plt.bar(x + 0.1, pred_euler, width=0.2, label="Pred")
    for i in range(3):
        plt.text(x[i] - 0.1, true_euler[i], f" {int(true_euler[i])}", ha="center", va="bottom")
        plt.text(x[i] + 0.1, pred_euler[i], f" {int(pred_euler[i])}", ha="center", va="bottom")
    plt.xticks(x, labels)
    plt.title("Euler Angles Comparison (Degrees)")
    plt.ylim(-10, 370)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_loss(recon, target, commit_loss, loss_weight):
    # 平移损失
    trans_loss = torch.sum(F.mse_loss(recon[:, :3], target[:, :3]))
    # 旋转损失（欧拉角直接MSE）
    rot_loss = torch.sum(F.mse_loss(recon[:, 3:], target[:, 3:]))
    commit_loss = torch.sum(commit_loss)
    total_loss = torch.sum(trans_loss * loss_weight[0] + rot_loss * loss_weight[1] + commit_loss * loss_weight[2])
    return {"total": total_loss, "trans": trans_loss, "rot": rot_loss, "commit": commit_loss}

def test_model(config):
    # 初始化模型
    model = VQRTCodec(config).to(device)
    model.load_state_dict(torch.load(config["test"]["test_model_path"]))
    model.eval()

    # 准备测试数据
    test_dataset = RTDataset(
        config["test"]["test_data_dir"],
        config["train"]["gcd_split_json"],
        config["train"]["gcd_root_dir"],
        'test'
    )
    test_loader = DataLoader(test_dataset, batch_size=config["test"]["batchsize"], shuffle=False)

    # 创建可视化目录
    vis_dir = os.path.join(config["test"]["visualization_dir"], "samples")
    os.makedirs(vis_dir, exist_ok=True)

    # 初始化统计量
    test_metrics = {"total": 0.0, "trans": 0.0, "rot": 0.0, "commit": 0.0}
    loss_weight = config["train"]["loss_weight"]

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            rt = batch["rt"].to(device)
            recon, _, _, commit_loss = model(rt)

            losses = calculate_loss(recon, rt, commit_loss, loss_weight)
            for k in test_metrics:
                test_metrics[k] += losses[k].item()

            # 可视化前几个样本
            if batch_idx == 0:
                batch_size = rt.size(0)
                for i in range(min(5, batch_size)):
                    idx = i + batch_idx * config["test"]["batchsize"]
                    true_rt = rt[i].cpu().numpy()
                    pred_rt = recon[i].cpu().numpy()
                    save_path = os.path.join(vis_dir, f"sample_{idx}.png")
                    visualize_sample(true_rt, pred_rt, save_path)

    # 计算平均损失
    for k in test_metrics:
        test_metrics[k] /= len(test_loader)

    print(
        f"""
============ Test Results ============
Total Loss: {test_metrics['total']:.4f}
Translation Loss: {test_metrics['trans']:.4f}
Rotation Loss: {test_metrics['rot']:.4f}
Commitment Loss: {test_metrics['commit']:.4f}
======================================
"""
    )
    print(f"Visualized samples saved to: {vis_dir}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Test Rotation-Translation VQ-VAE")
    # parser.add_argument("--config_path", type=str, required=True, help="Path to config YAML")
    # args = parser.parse_args()
    config = read_yaml_config("/data/qj/codec_codebook/train_rt/config_rt_euler.yaml")
    test_model(config)
