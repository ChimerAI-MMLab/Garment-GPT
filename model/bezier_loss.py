import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DynamicLossWeight:
    def __init__(self, num_losses, alpha=0.8, min_weight=0.1, max_weight=2.0):
        """
        alpha: 平滑系数 (0 < alpha < 1)
        min_weight/max_weight: 权重裁剪范围
        """
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.ema_values = torch.ones(num_losses)  # 初始EMA值
        self.normalization = None  # 初始标准化基准

    def __call__(self, current_losses):
        # 转换为Tensor
        current_losses = torch.tensor(current_losses)

        # 初始化标准化基准
        if self.normalization is None:
            self.normalization = current_losses.mean(dim=0).detach()

        # 更新EMA (指数移动平均)
        self.ema_values = self.alpha * self.ema_values + (1 - self.alpha) * current_losses.detach()

        # 标准化损失值
        normalized_losses = current_losses / (self.ema_values + 1e-8)

        # 计算动态权重
        weights = 1.0 / (normalized_losses + 1e-8)

        # 权重归一化
        weights = weights / weights.sum()

        # 权重裁剪
        weights = torch.clamp(weights, self.min_weight, self.max_weight)

        return weights.numpy()


def compute_loss(pred_control_points, bezier_curve, sample_points, writer=None,is_sum_vertices_loss=True):
    # 计算原始数据点参数t（基于弦长）

    # original_diff = torch.diff(target_points, dim=0)
    # original_chord_lengths = torch.norm(original_diff, dim=1)
    # t_original = torch.cat([torch.zeros([1], device=device), torch.cumsum(original_chord_lengths, dim=0)])
    # t_original = t_original / t_original[-1]  # 归一化到[0,1]

    # 计算sample数据点参数t（基于弦长）
    sample_diff = torch.diff(sample_points, dim=-2)
    sample_chord_lengths = torch.norm(sample_diff, dim=-1)
    t_sample = torch.cat([torch.zeros(sample_points.shape[0], 1, device=sample_chord_lengths.device, dtype=torch.float32), torch.cumsum(sample_chord_lengths, dim=-1)], dim=-1)
    t_sample = t_sample / t_sample[:, -1:]  # 归一化到[0,1]
    target_end_points = sample_points[:, [0, -1], :]
    # pred_points = bezier_curve(t_original)
    pred_sample_points = bezier_curve(t_sample)

    intermediate_point_loss = torch.norm(pred_sample_points - sample_points, dim=-1, p=2)
    
    if is_sum_vertices_loss:
        vertices_loss = torch.sum(torch.norm(pred_control_points[:, [0, -1], :] - target_end_points, dim=-1, p=2))
    else:
        vertices_loss = torch.norm(pred_control_points[:, [0, -1], :] - target_end_points, dim=-1, p=2)

    # # ========== 最近点匹配损失 ==========
    # # 在曲线上密集采样
    # dense_t = torch.linspace(0, 1, 1000, device=target_points.device)  # 采样1000个点
    # dense_points = bezier_curve(dense_t)  # (1000, 2)

    # # 计算所有原始点与密集采样点的距离矩阵
    # # target_points: (N,2), dense_points: (M,2)
    # diff = target_points.unsqueeze(1) - dense_points.unsqueeze(0)  # (N,M,2)
    # distances = torch.norm(diff, dim=2, p=2)  # (N,M)

    # # 找到每个原始点的最近曲线点
    # min_distances, _ = torch.min(distances, dim=1)  # (N,)

    # # 计算最近点损失
    # nearest_point_loss = torch.sum(min_distances)

    # 平滑性约束（相邻控制点间距）
    # diff = torch.diff(control_points, dim=0)
    # smooth_loss = 0.01 * torch.sum(torch.norm(diff, dim=1))

    # dynamic_weights = DynamicLossWeight(num_losses=2)([
    #     intermediate_point_loss.item(),
    #     vertices_loss.item()
    # ])

    # # 应用动态权重
    # total_loss = (
    #     dynamic_weights[0] * intermediate_point_loss +
    #     dynamic_weights[1] * vertices_loss
    # )

    return intermediate_point_loss, vertices_loss
