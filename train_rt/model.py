import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ, FSQ, ResidualFSQ


class RTEncoder(nn.Module):
    """旋转平移编码器"""

    def __init__(self, input_dim=7, latent_dim=64, hidden_dim=256, num_layers=4):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)

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
        self.output_fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.input_fc(x)
        for block in self.blocks:
            residual = x
            x = block(x) + residual
            x = F.gelu(x)
        return self.output_fc(x)


class RTDecoder(nn.Module):
    """旋转平移解码器"""

    def __init__(self, latent_dim=64, output_dim=7, hidden_dim=256, num_layers=4):
        super().__init__()
        self.input_fc = nn.Linear(latent_dim, hidden_dim)

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
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_fc(x)
        for block in self.blocks:
            residual = x
            x = block(x) + residual
            x = F.gelu(x)
        return self.output_fc(x)


class RTCodebook(nn.Module):
    def __init__(self, latent_dim=64, codebook_size=512, num_quantizers=2, decay=0.99, is_fsq=False, fsq_levels=None):
        super().__init__()
        if not is_fsq:
            self.quantizer = ResidualVQ(
                dim=latent_dim, codebook_size=codebook_size, num_quantizers=num_quantizers, shared_codebook=True, decay=decay, stochastic_sample_codes=True, sample_codebook_temp=0.1
            )
        else:
            self.quantizer = FSQ(levels=fsq_levels)

    def forward(self, z):
        if isinstance(self.quantizer, ResidualVQ):
            quantized, indices, commit_loss = self.quantizer(z)
            return quantized, indices, commit_loss
        else:
            quantized, indices = self.quantizer(z)
            return quantized, indices, 0.0


class VQRTCodec(nn.Module):

    def __init__(self, config):
        super().__init__()
        cfg = config["rt_model"]

        self.encoder = RTEncoder(input_dim=cfg["input_dim"], latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"])

        self.codebook = RTCodebook(
            latent_dim=cfg["latent_dim"],
            codebook_size=cfg["codebook"]["size"],
            num_quantizers=cfg["codebook"]["num_quantizers"],
            decay=cfg["codebook"]["decay"],
            is_fsq=cfg["codebook"]["is_fsq"],
            fsq_levels=cfg["codebook"]["fsq_levels"],
        )

        self.decoder = RTDecoder(latent_dim=cfg["latent_dim"], output_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"])

    def forward(self, x):
        z = self.encoder(x)
        quantized, indices, commit_loss = self.codebook(z)
        recon = self.decoder(quantized)
        return recon, quantized, indices, commit_loss
