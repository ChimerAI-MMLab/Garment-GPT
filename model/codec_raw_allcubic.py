import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch import ResidualVQ, ResidualFSQ, FSQ

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
    """基于 ResNet 的多段线解码器，输出线型分类和对应参数"""

    def __init__(self, latent_dim, hidden_dim=256, num_layers=4):
        super(ResNetDecoder_wClassifier, self).__init__()

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

        # 线型分类器：[直线，二次贝塞尔，三次贝塞尔，圆弧]
        self.line_type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),  # 3种线型 两个贝塞尔合并为三次贝塞尔一种
        )

        # 输出参数：[起点(2), 终点(2), 二次控制点(2), 三次控制点(2), 圆弧半径(1), 圆弧优劣(1), 圆弧方向(1)]
        # 总共13个参数
        self.params_fc = nn.Linear(hidden_dim, 11)

    def forward(self, x):
        # x: (bs, latent_dim)
        x_hidden = self.input_fc(x)  # (bs, hidden_dim)

        for block in self.blocks:
            residual = x_hidden
            x_hidden = block(x_hidden) + residual
            x_hidden = F.gelu(x_hidden)
        line_type_logits = self.line_type_classifier(x_hidden)  # (bs, 4)
        params = self.params_fc(x_hidden)  # (bs, 13)
        
        # 分解参数
        start_point = params[:, 0:2]  # (bs, 2)
        end_point = params[:, 2:4]  # (bs, 2)
        # quad_control_point = params[:, 4:6]  # (bs, 2) 二次贝塞尔控制点
        cubic_control_point1 = params[:, 4:6]  # (bs, 2) 三次贝塞尔第一个控制点
        cubic_control_point2 = params[:, 6:8]  # (bs, 2) 三次贝塞尔第二个控制点
        arc_radius = params[:, 8:9]  # (bs, 1)
        arc_large_flag = torch.sigmoid(params[:, 9:10])  # (bs, 1) 圆弧优劣标志
        arc_sweep_flag = torch.sigmoid(params[:, 10:11])  # (bs, 1) 圆弧方向标志

        return {
            'line_type_logits': line_type_logits,
            'start_point': start_point,
            'end_point': end_point,
            # 'quad_control_point': quad_control_point,
            'cubic_control_point1': cubic_control_point1,
            'cubic_control_point2': cubic_control_point2,
            'arc_radius': arc_radius,
            'arc_large_flag': arc_large_flag,
            'arc_sweep_flag': arc_sweep_flag
        }


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
    def __init__(self, input_dim, latent_dim, hidden_dim, N, num_layers, n_heads, codebook_size, num_quantizers, decay, is_fsq, fsq_levels):
        super(VQAutoCodec, self).__init__()
        self.encoder = ResNetEncoder(input_dim, latent_dim, N, hidden_dim, num_layers)
        self.codebook = Codebook(codebook_size, latent_dim, num_quantizers, decay, is_fsq, fsq_levels)
        self.decoder = ResNetDecoder_wClassifier(latent_dim, hidden_dim, num_layers)

    def forward(self, x):
        z = self.encoder(x)  # 编码器输出潜在嵌入
        quantized, indices, commit_loss = self.codebook(z)  # 使用 ResidualVQ 进行量化
        decoder_output = self.decoder(quantized)  # 解码器恢复数据
        return decoder_output, quantized, indices, z, commit_loss


# 新增：Transformer编码器 + ResNet解码器的codec
class TransformerEncoder(nn.Module):
    """基于Transformer的多段线编码器，输入32个采样点，每点2维"""
    def __init__(self, input_dim, latent_dim, num_points=32, hidden_dim=256, num_layers=4, n_heads=8):
        super(TransformerEncoder, self).__init__()
        self.num_points = num_points
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*2, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(num_points * hidden_dim, latent_dim)

    def forward(self, x):
        # x: (bs, num_points, 2)
        bs = x.shape[0]
        x = self.input_fc(x)  # (bs, num_points, hidden_dim)
        x = self.transformer(x)  # (bs, num_points, hidden_dim)
        x = x.reshape(bs, -1)  # (bs, num_points*hidden_dim)
        return self.output_fc(x)  # (bs, latent_dim)


class VQAutoCodec_TransformerEncoder(nn.Module):
    """
    编码器用Transformer，解码器用ResNet，codebook和参数与原版一致。
    Transformer输入32个采样点，每点2维。
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, N, num_layers, n_heads, codebook_size, num_quantizers, decay, is_fsq, fsq_levels):
        super(VQAutoCodec_TransformerEncoder, self).__init__()
        self.encoder = TransformerEncoder(input_dim, latent_dim, num_points=N, hidden_dim=hidden_dim, num_layers=num_layers, n_heads=n_heads)
        self.codebook = Codebook(codebook_size, latent_dim, num_quantizers, decay, is_fsq, fsq_levels)
        self.decoder = ResNetDecoder_wClassifier(latent_dim, hidden_dim, num_layers)

    def forward(self, x):
        # x: (bs, 32, 2)
        z = self.encoder(x)
        quantized, indices, commit_loss = self.codebook(z)
        decoder_output = self.decoder(quantized)
        return decoder_output, quantized, indices, z, commit_loss


if __name__ == "__main__":
    # 原ResNet编码器+解码器测试
    # 测试VQAutoCodec
    codec = VQAutoCodec(
        input_dim=2,
        latent_dim=128,
        hidden_dim=256,
        N=32,
        num_layers=4,
        codebook_size=512,
        num_quantizers=2,
        n_heads=8,
        decay=0.99,
        is_fsq=False,
        fsq_levels=8,
    )
    batch = torch.randn(32, 32, 2)  # (bs=32, N=32, 2)
    decoder_output, quantized, indices, z, commit_loss = codec(batch)
    print("VQAutoCodec 输入形状:", batch.shape)
    print("VQAutoCodec 编码后潜在向量形状:", z.shape)
    print("VQAutoCodec 解码输出keys:", decoder_output.keys())

    # 测试VQAutoCodec_TransformerEncoder
    codec_trans = VQAutoCodec_TransformerEncoder(
        input_dim=2,
        latent_dim=128,
        hidden_dim=256,
        N=32,
        num_layers=4,
        n_heads=8,
        codebook_size=512,
        num_quantizers=2,
        decay=0.99,
        is_fsq=False,
        fsq_levels=8,
    )
    batch32 = torch.randn(16, 32, 2)  # (bs=16, 32, 2)
    decoder_output2, quantized2, indices2, z2, commit_loss2 = codec_trans(batch32)
    print("VQAutoCodec_TransformerEncoder 输入形状:", batch32.shape)
    print("VQAutoCodec_TransformerEncoder 编码后潜在向量形状:", z2.shape)
    print("VQAutoCodec_TransformerEncoder 解码输出keys:", decoder_output2.keys())
