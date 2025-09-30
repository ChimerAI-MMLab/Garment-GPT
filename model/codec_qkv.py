import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ, FSQ


class PositionalEncoding(nn.Module):
    """动态可学习位置编码"""

    def __init__(self, d_model, max_len):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe


class TransformerEncoder(nn.Module):
    """改进的Transformer编码器（自动维度校正）"""

    def __init__(self, input_dim, code_dim, num_points, hidden_dim=256, num_layers=6, nhead=8):
        super().__init__()
        # 输入特征映射
        self.embed = nn.Linear(input_dim, hidden_dim)

        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=num_points)

        # Transformer结构
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, batch_first=True), num_layers=num_layers)

        # 维度校正层（在展平前调整每个位置的维度）
        self.proj = nn.Linear(hidden_dim, code_dim // num_points)

    def forward(self, x):
        # x: (bs, num_points, input_dim)
        x = self.embed(x)  # (bs, N, h)
        x = self.pos_encoder(x)  # 添加位置编码
        x = self.encoder(x)  # (bs, N, h)
        x = self.proj(x)  # (bs, N, code_dim//N)
        return x.flatten(1)  # (bs, code_dim)


class TransformerDecoder_wClassifier(nn.Module):

    def __init__(self, code_dim, num_points, hidden_dim, query_size, num_layers, nhead):
        super().__init__()
        # 重塑参数
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim

        # 可学习查询向量
        self.query = nn.Parameter(torch.randn(query_size, hidden_dim) * 0.1)

        # Transformer解码器
        self.kv_adapter = nn.Linear(self.code_dim // self.num_points, self.hidden_dim)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, batch_first=True), num_layers=num_layers)

        # 输出投影
        self.coord_proj = nn.Linear(hidden_dim, 2)

        # 分类器
        self.classifier = nn.Sequential(nn.Linear(code_dim, code_dim // 2), nn.GELU(), nn.LayerNorm(code_dim // 2), nn.Linear(code_dim // 2, 1), nn.Sigmoid())

    def forward(self, codes):
        # codes: (bs, code_dim)
        bs = codes.size(0)

        # 重塑为序列格式 (bs, num_points, code_dim // num_points)
        kv = codes.view(bs, self.num_points, self.code_dim // self.num_points)
        kv = self.kv_adapter(kv)

        # 生成查询 (扩展至batch维度)
        queries = self.query.unsqueeze(0).expand(bs, -1, -1)  # (bs, 3, h)

        # 解码过程
        decoded = self.decoder(queries, kv)
        coords = self.coord_proj(decoded)  # (bs, 3, 2)

        # 分类结果
        is_line = self.classifier(codes)  # (bs, 1)

        return coords, is_line


class VQTransformer(nn.Module):
    """完整模型架构"""

    def __init__(self, input_dim, code_dim, hidden_dim, num_points, num_layers, nheads, codebook_size, num_control_points, num_quantizers, decay, is_fsq, fsq_levels):
        super().__init__()
        self.is_fsq = is_fsq
        self.encoder = TransformerEncoder(input_dim=input_dim, code_dim=code_dim, num_points=num_points, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nheads)
        if not is_fsq:
            self.codebook = ResidualVQ(
                dim=code_dim, codebook_size=codebook_size, num_quantizers=num_quantizers, shared_codebook=True, decay=decay, stochastic_sample_codes=True, sample_codebook_temp=0.1
            )
        else:
            self.codebook = FSQ(dim=code_dim, levels=fsq_levels, channel_first=True)

        self.decoder = TransformerDecoder_wClassifier(code_dim=code_dim, num_points=num_points, hidden_dim=hidden_dim, query_size=num_control_points, num_layers=num_layers, nhead=nheads)

    def forward(self, x):
        z = self.encoder(x)  # (bs, code_dim)

        if not self.is_fsq:
            quantized, indices, commit_loss = self.codebook(z)
        else:
            quantized, indices = self.codebook(z)
            commit_loss = None
        coords, is_line = self.decoder(quantized)
        return coords, quantized, indices, z, is_line, commit_loss


# 测试用例
if __name__ == "__main__":
    # 示例配置：num_points=32, code_dim=256 ⇒ hidden_dim=8 (256//32=8)
    model = VQTransformer(input_dim=2, code_dim=256, num_points=32, hidden_dim=8)

    # 测试输入
    test_input = torch.randn(4, 32, 2)  # (bs, num_points, input_dim)
    coords, quantized, indices, z, is_line, commit_loss = model(test_input)

    print(f"输入形状: {test_input.shape}")
    print(f"坐标输出: {coords.shape}")  # 预期: (4, 3, 2)
    print(f"分类输出: {is_line.shape}")  # 预期: (4, 1)
