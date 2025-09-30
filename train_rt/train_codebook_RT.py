import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import yaml
from dataset_RT_GCD import RTDataset
from model import VQRTCodec

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_yaml_config(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def train_rt_vqvae(config_path):
    # 初始化配置
    config = read_yaml_config(config_path)

    # 创建输出目录
    os.makedirs(config["train"]["model_save_dir"], exist_ok=True)
    os.makedirs(config["test"]["indices_distribution_dir"], exist_ok=True)

    # 准备数据集
    train_dataset = RTDataset(config["train"]["train_data_dir"], config["train"]["gcd_split_json"], config["train"]["gcd_root_dir"],'train')
    test_dataset = RTDataset(config["test"]["test_data_dir"], config["train"]["gcd_split_json"], config["train"]["gcd_root_dir"],'test')

    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batchsize"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["test"]["batchsize"], shuffle=False)

    # 初始化模型
    model = VQRTCodec(config).to(device)
    opt = optim.Adam(model.parameters(), lr=config["train"]["lr"])

    # 日志记录
    writer = SummaryWriter(log_dir=config["tensorboard_log_dir"])

    # 混合损失函数
    def calculate_loss(recon, target, commit_loss):
        # 平移损失
        trans_loss = torch.sum(F.mse_loss(recon[:, :3], target[:, :3]))

        # 旋转损失
        # q_pred = F.normalize(recon[:, 3:], dim=1)
        # q_true = F.normalize(target[:, 3:], dim=1)
        # dot = torch.sum(q_pred * q_true, dim=1)
        # dot = torch.clamp(dot, -1 + 1e-6, 1 - 1e-6)
        # rot_loss = torch.sum(1 - torch.abs(dot))
        rot_loss = torch.sum(F.mse_loss(recon[:, 3:], target[:, 3:]))

        commit_loss = torch.sum(commit_loss)

        loss_weight = config["train"]["loss_weight"]
        total_loss = torch.sum(trans_loss * loss_weight[0] + rot_loss * loss_weight[1] + commit_loss * loss_weight[2])

        return {"total": total_loss, "trans": trans_loss, "rot": rot_loss, "commit": commit_loss}

    for epoch in range(config["train"]["epochs"]):
        # 训练阶段
        model.train()
        train_metrics = {"total": 0.0, "trans": 0.0, "rot": 0.0, "commit": 0.0}

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            rt = batch["rt"].to(device)

            opt.zero_grad()
            recon, _, indices, commit_loss = model(rt)

            losses = calculate_loss(recon, rt, commit_loss)
            losses["total"].backward()
            opt.step()

            for k in train_metrics:
                train_metrics[k] += losses[k].item()

        # 计算平均训练损失
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
            writer.add_scalar(f"Train/{k}", train_metrics[k], epoch)

        # 测试阶段
        model.eval()
        test_metrics = {"total": 0.0, "trans": 0.0, "rot": 0.0, "commit": 0.0}
        all_indices = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Test Epoch {epoch+1}"):
                rt = batch["rt"].to(device)
                recon, _, indices, commit_loss = model(rt)

                losses = calculate_loss(recon, rt, commit_loss)

                for k in test_metrics:
                    test_metrics[k] += losses[k].item()

                # 收集索引数据
                all_indices.extend(list(indices.cpu().numpy().flatten()))

        # 处理测试结果
        for k in test_metrics:
            test_metrics[k] /= len(test_loader)
            writer.add_scalar(f"Test/{k}", test_metrics[k], epoch)

        torch.save(model.state_dict(), os.path.join(config["train"]["model_save_dir"], f"epoch_{epoch+1}_{test_metrics['total']:.4f}.pth"))

        unique_indices = np.unique(all_indices)
        codebook_size = config["rt_model"]["codebook"]["size"]
        utilization_rate = len(unique_indices) / codebook_size * 100

        # 绘制带利用率信息的直方图
        plt.figure(figsize=(12, 6))
        plt.hist(all_indices, bins=codebook_size, range=(0, codebook_size))
        plt.title(f"Codebook Indices Distribution (Epoch {epoch+1})\n" f"Utilization: {utilization_rate:.1f}% ({len(unique_indices)}/{codebook_size})")
        plt.xlabel("Codebook Index")
        plt.ylabel("Count")

        # 添加文本标注
        text_str = f"Top 5 Used Indices:\n"
        values, counts = np.unique(all_indices, return_counts=True)
        top5_idx = values[np.argsort(-counts)[:5]]
        for idx in top5_idx:
            text_str += f"Index {int(idx)}: {counts[values==idx][0]} samples\n"

        plt.annotate(text_str, xy=(0.65, 0.7), xycoords="axes fraction", fontsize=9, bbox=dict(boxstyle="round", alpha=0.1))

        plt.savefig(os.path.join(config["test"]["indices_distribution_dir"], f"epoch_{epoch+1}.png"))
        plt.close()

        log_msg = f"Train Loss: {train_metrics['total']:.4f} (Trans: {train_metrics['trans']:.4f}, Rot: {train_metrics['rot']:.4f})\nTest Loss: {test_metrics['total']:.4f} (Trans: {test_metrics['trans']:.4f}, Rot: {test_metrics['rot']:.4f})"
        print(log_msg)

        # 记录当前学习率
        current_lr = opt.param_groups[0]['lr']
        writer.add_scalar("LR", current_lr, epoch)

    writer.close()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Training script with config path.")
    # parser.add_argument("--config_path", type=str, help="Path to the configuration YAML file.")
    # args = parser.parse_args()
    # config_path = args.config_path
    config_path = "/data/qj/codec_codebook/train_rt/config_rt.yaml"
    train_rt_vqvae(config_path)
