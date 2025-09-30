import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch import ResidualVQ, ResidualFSQ, FSQ


# class ResidualVQ(nn.Module):
#     def __init__(self, dim, codebook_size, num_quantizers=2, use_ema=False, decay=0.99, eps=1e-5):
#         super().__init__()
#         self.dim = dim
#         self.codebook_size = codebook_size
#         self.num_quantizers = num_quantizers
#         self.use_ema = use_ema
#         self.decay = decay
#         self.eps = eps

#         # 初始化量化器列表（支持残差级联）
#         self.layers = nn.ModuleList()
#         for _ in range(num_quantizers):
#             if use_ema:
#                 quantizer = EMAQuantizer(dim, codebook_size, decay, eps)
#             else:
#                 quantizer = CodebookLossQuantizer(dim, codebook_size)
#             self.layers.append(quantizer)

#     def forward(self, x):
#         residual = x
#         all_indices = []
#         total_cmt_loss = 0.0
#         total_cb_loss = 0.0

#         for quantizer in self.layers:
#             # 量化当前残差
#             quantized, indices, cmt_loss, cb_loss = quantizer(residual)
#             # 更新残差（保持梯度流）
#             residual = residual - quantized.detach()
#             all_indices.append(indices)
#             total_cmt_loss += cmt_loss
#             if not self.use_ema:
#                 total_cb_loss += cb_loss
#         if self.use_ema:
#             total_cb_loss = None
#         # 最终量化结果是所有量化器输出的和
#         quantized_out = x - residual
#         indices = torch.stack(all_indices, dim=-1)
#         return quantized_out, indices, total_cmt_loss, total_cb_loss


class CodebookLossQuantizer(nn.Module):
    """基于codebook_loss的梯度更新方式"""

    def __init__(self, dim, codebook_size):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(codebook_size, dim))
        nn.init.kaiming_uniform_(self.embedding, mode="fan_in")

    def forward(self, x):
        # 计算欧式距离
        distances = torch.sum(x**2, dim=-1, keepdim=True) - 2 * torch.matmul(x, self.embedding.t()) + torch.sum(self.embedding.t() ** 2, dim=0, keepdim=True)

        # 寻找最近邻索引
        indices = torch.argmax(-distances, dim=-1)
        quantized = F.embedding(indices, self.embedding)

        # 计算两种损失
        codebook_loss = F.mse_loss(quantized, x.detach())  # 将codebook向量拉向encoder输出
        commitment_loss = F.mse_loss(x, quantized.detach())  # 将encoder输出拉向codebook

        # 直通估计器（Straight-Through Estimator）
        quantized = x + (quantized - x).detach()
        return quantized, indices, commitment_loss, codebook_loss


class EMAQuantizer(nn.Module):

    def __init__(self, dim, codebook_size, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps

        # 初始化codebook和相关统计量
        self.register_buffer("embedding", torch.randn(codebook_size, dim))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embedding_avg", self.embedding.clone())

    def forward(self, x):
        # 展平输入（支持任意维度）
        input_shape = x.shape
        x_flat = x.view(-1, self.embedding.size(1))

        # 计算距离并分配索引
        distances = torch.sum(x_flat**2, dim=1, keepdim=True) - 2 * torch.matmul(x_flat, self.embedding.t()) + torch.sum(self.embedding.t() ** 2, dim=0, keepdim=True)
        indices = torch.argmax(-distances, dim=-1)
        encodings = F.one_hot(indices, self.embedding.size(0)).float()

        # EMA更新（仅在训练时更新）
        if self.training:
            # 更新聚类统计量
            cluster_size = encodings.sum(0)
            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            # 更新嵌入均值
            embed_sum = torch.matmul(encodings.t(), x_flat)
            self.embedding_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # 标准化更新嵌入
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.embedding.size(0) * self.eps) * n
            self.embedding.data.copy_(self.embedding_avg / cluster_size.unsqueeze(1))

        # 获取量化结果
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view(*input_shape)

        # 计算commitment loss
        commitment_loss = F.mse_loss(x, quantized.detach())

        quantized = x + (quantized - x).detach()
        return quantized, indices, commitment_loss, None


class ResNetEncoder(nn.Module):
    """基于 ResNet 的多段线编码器"""

    def __init__(self, input_dim, latent_dim, num_points, hidden_dim=256, num_layers=4):
        super(ResNetEncoder, self).__init__()
        self.num_points = num_points

        # 输入特征投影
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # 残差块堆叠
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        # 输出映射到潜在空间
        self.output_fc = nn.Linear(num_points * hidden_dim, latent_dim)

    def forward(self, x):
        # x: (bs, num_points, 2)
        bs = x.shape[0]
        x = self.input_fc(x)  # (bs, num_points, hidden_dim)

        # 残差处理
        for block in self.blocks:
            residual = x
            x = block(x) + residual  # 残差连接
            x = F.gelu(x)

        # 展平并映射到潜在空间
        x = x.reshape(bs, -1)  # (bs, num_points*hidden_dim)
        return self.output_fc(x)  # (bs, latent_dim)


class ResNetDecoder_wClassifier(nn.Module):
    """基于 ResNet 的多段线解码器，额外输出是否为直线的分类结果"""

    def __init__(self, latent_dim, num_points, hidden_dim=256, num_layers=4):
        super(ResNetDecoder_wClassifier, self).__init__()
        self.num_points = num_points

        # 输入映射到隐藏维度
        self.input_fc = nn.Linear(latent_dim, hidden_dim)

        # 残差块堆叠
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        # 输出映射到坐标空间
        self.output_fc = nn.Linear(hidden_dim, 2 * num_points)

        # 是否为直线的分类器
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid(),  # 输出概率值
        )

    def forward(self, x):
        # x: (bs, latent_dim)
        is_line = self.classifier(x)  # (bs, 1)

        x = self.input_fc(x)  # (bs, hidden_dim)

        for block in self.blocks:
            residual = x
            x = block(x) + residual
            x = F.gelu(x)

        coordinates = self.output_fc(x).reshape(x.size(0), self.num_points, 2)  # (bs, num_points, 2)

        return coordinates, is_line


class ResNetDecoder(nn.Module):
    """基于 ResNet 的多段线解码器"""

    def __init__(self, latent_dim, num_points, hidden_dim=256, num_layers=4):
        super(ResNetDecoder, self).__init__()
        self.num_points = num_points

        # 输入映射到隐藏维度
        self.input_fc = nn.Linear(latent_dim, hidden_dim)

        # 残差块堆叠
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        self.output_fc = nn.Linear(hidden_dim, 2 * num_points)

    def forward(self, x):
        # x: (bs, latent_dim)
        x = self.input_fc(x)  # (bs, hidden_dim)

        for block in self.blocks:
            residual = x
            x = block(x) + residual
            x = F.gelu(x)

        x = self.output_fc(x).reshape(x.size(0), self.num_points, 2)  # (bs, num_points, 2)
        return x


class Codebook(nn.Module):
    def __init__(self, codebook_size, embedding_dim, num_quantizers, decay, is_fsq, fsq_levels):
        super(Codebook, self).__init__()
        self.is_fsq = is_fsq
        if not is_fsq:
            self.residual_vq = ResidualVQ(
                dim=embedding_dim, codebook_size=codebook_size, num_quantizers=num_quantizers, shared_codebook=True, decay=decay, stochastic_sample_codes=True, sample_codebook_temp=0.1
            )
        else:
            self.residual_vq = FSQ(dim=embedding_dim, levels=fsq_levels, channel_first=True)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if not self.is_fsq:
            quantized, indices, commit_loss = self.residual_vq(x)
        else:
            quantized, indices = self.residual_vq(x)
            commit_loss = None
        return quantized, indices, commit_loss


class VQAutoCodec(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, N, num_layers, n_heads, codebook_size, num_control_points, num_quantizers, decay, is_fsq, fsq_levels):
        super(VQAutoCodec, self).__init__()
        self.encoder = ResNetEncoder(input_dim, latent_dim, N, hidden_dim, num_layers)
        self.codebook = Codebook(codebook_size, latent_dim, num_quantizers, decay, is_fsq, fsq_levels)
        self.decoder = ResNetDecoder_wClassifier(latent_dim, num_control_points, hidden_dim, num_layers)

    def forward(self, x):
        z = self.encoder(x)  # 编码器输出潜在嵌入
        quantized, indices, commit_loss = self.codebook(z)  # 使用 ResidualVQ 进行量化
        coordinates, is_line = self.decoder(quantized)  # 解码器恢复数据
        return coordinates, quantized, indices, z, is_line, commit_loss


if __name__ == "__main__":
    encoder = ResNetEncoder(input_dim=2, N=100, latent_dim=128)
    decoder = ResNetDecoder(latent_dim=128, num_points=10)
    batch = torch.randn(32, 100, 2)  # (bs=32, N=100, 2)

    latent = encoder(batch)  # (32, 128)
    reconstructed = decoder(latent)  # (32, 10, 2)

    print("输入形状:", batch.shape)
    print("编码后潜在向量形状:", latent.shape)
    print("解码重建坐标形状:", reconstructed.shape)
