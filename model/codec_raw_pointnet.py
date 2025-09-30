import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch import ResidualVQ, ResidualFSQ, FSQ

# 引入PointNet编码器
from model.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model as PointNetEncoderRaw

class PointNetEncoder(nn.Module):
    """
    基于PointNet的多段线编码器
    输入: (bs, N, 2) 或 (bs, N, 3)
    输出: (bs, latent_dim)
    """
    def __init__(self, input_dim, latent_dim, N=32, use_normals=False):
        super(PointNetEncoder, self).__init__()
        # PointNet输入为 (B, C, N)
        self.input_dim = input_dim
        self.N = N
        self.use_normals = use_normals
        # PointNet输出1024维
        self.pointnet = PointNetEncoderRaw(num_class=latent_dim, normal_channel=use_normals)
        # 线性层将PointNet输出映射到latent_dim
        self.fc = nn.Linear(1024, latent_dim)

    def forward(self, x):
        # x: (bs, N, input_dim)
        # 输入的xy坐标不变，z坐标统一为0
        if x.dim() == 2:
            x = x.unsqueeze(0)
        bs = x.shape[0]
        N = x.shape[1]
        # 保证输入为 (bs, N, 2) 或 (bs, N, 3)
        if x.shape[-1] == 2:
            # (bs, N, 2) -> (bs, N, 3), z=0
            zeros = torch.zeros(bs, N, 1, dtype=x.dtype, device=x.device)
            x = torch.cat([x, zeros], dim=-1)
        # (bs, N, 3) -> (bs, 3, N)
        x = x.permute(0, 2, 1)
        # PointNet返回: x(1024), logits, l3_points
        x_feat, _, _ = self.pointnet(x)
        z = self.fc(x_feat)
        # z = x_feat
        return z  # (bs, latent_dim)

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
            nn.Linear(hidden_dim // 2, 4),  # 4种线型
        )

        # 输出参数：[起点(2), 终点(2), 二次控制点(2), 二次控制点(2), 三次控制点(2), 圆弧半径(1), 圆弧优劣(1), 圆弧方向(1)]
        # 总共13个参数
        self.params_fc = nn.Linear(hidden_dim, 13)

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
        quad_control_point = params[:, 4:6]  # (bs, 2) 二次贝塞尔控制点
        cubic_control_point1 = params[:, 6:8]  # (bs, 2) 三次贝塞尔第一个控制点
        cubic_control_point2 = params[:, 8:10]  # (bs, 2) 三次贝塞尔第二个控制点
        arc_radius = params[:, 10:11]  # (bs, 1)
        arc_large_flag = torch.sigmoid(params[:, 11:12])  # (bs, 1) 圆弧优劣标志
        arc_sweep_flag = torch.sigmoid(params[:, 12:13])  # (bs, 1) 圆弧方向标志

        return {
            'line_type_logits': line_type_logits,
            'start_point': start_point,
            'end_point': end_point,
            'quad_control_point': quad_control_point,
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


class VQAutoCodec_PointNet(nn.Module):
    """
    编码器用PointNet，解码器用ResNet，codebook和参数与原版一致。
    输入: (bs, N, 2) 或 (bs, N, 3)
    """
    def __init__(self, input_dim, latent_dim, hidden_dim, N, num_layers, n_heads, codebook_size, num_quantizers, decay, is_fsq, fsq_levels, use_normals=False):
        super(VQAutoCodec_PointNet, self).__init__()
        self.encoder = PointNetEncoder(input_dim, latent_dim, N=N, use_normals=use_normals)
        self.codebook = Codebook(codebook_size, latent_dim, num_quantizers, decay, is_fsq, fsq_levels)
        self.decoder = ResNetDecoder_wClassifier(latent_dim, hidden_dim, num_layers)

    def forward(self, x):
        z = self.encoder(x)  # 编码器输出潜在嵌入
        quantized, indices, commit_loss = self.codebook(z)  # 使用 ResidualVQ 进行量化
        decoder_output = self.decoder(quantized)  # 解码器恢复数据
        return decoder_output, quantized, indices, z, commit_loss


if __name__ == "__main__":
    # 测试VQAutoCodec_PointNet
    codec_pointnet = VQAutoCodec_PointNet(
        input_dim=2,
        latent_dim=1024,
        hidden_dim=256,
        N=1024,
        num_layers=4,
        codebook_size=1024,
        num_quantizers=2,
        n_heads=8,
        decay=0.99,
        is_fsq=False,
        fsq_levels=8,
        use_normals=False,
    )
    batch = torch.randn(16, 1024, 2)  # (bs=16, N=32, 2)

    checkpoint_path = "/data/qj/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 获取 state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 修改键名，添加缺少的前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        # 添加 encoder.pointnet. 前缀
        new_key = f"encoder.pointnet.{key}"
        new_state_dict[new_key] = value

    # 加载修改后的 state_dict
    codec_pointnet.load_state_dict(new_state_dict, strict=False)
    codec_pointnet.eval()
    decoder_output, quantized, indices, z, commit_loss = codec_pointnet(batch)
    torch.set_printoptions(threshold=float('inf'), linewidth=200, sci_mode=False)
    print("VQAutoCodec_PointNet 输入形状:", batch.shape)
    print("VQAutoCodec_PointNet 编码后潜在向量形状:", z.shape)
    print("VQAutoCodec_PointNet 解码输出keys:", decoder_output.keys())
