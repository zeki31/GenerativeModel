import torch
from loguru import logger
from torch import nn


def _pos_encoding(t, output_dim, device="cpu"):
    v = torch.zeros(output_dim, device=device)

    i = torch.arange(0, output_dim, device=device)
    div_term = 10000 ** (i / output_dim)

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])

    return v


def pos_encoding(ts, output_dim, device="cpu"):
    batch_size = len(ts)
    v = torch.zeros(batch_size, output_dim, device=device)
    for t in range(batch_size):
        v[t] = _pos_encoding(t=ts[t], output_dim=output_dim, device=device)
    return v


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch),
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y


class UNet(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100, num_labels=None):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, time_embed_dim)

    def forward(self, x, timesteps, labels=None):
        # Positional encoding for timesteps
        v = pos_encoding(timesteps, output_dim=self.time_embed_dim, device=x.device)

        if labels is not None:
            v += self.label_emb(labels)

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)
        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x


if __name__ == "__main__":
    model = UNet(in_ch=1, time_embed_dim=16)
    x = torch.randn(2, 1, 28, 28)  # Batch size of 2, 1 channel, 28x28 image
    y = model(x, timesteps=[1.0, 2.0])
    logger.info(f"Output shape: {y.shape}")  # Should be (2, 1, 28, 28)

    ts = [1.0, 2.0]
    v = pos_encoding(ts, output_dim=16)
    for t, encoding in zip(ts, v):
        logger.info(f"Positional encoding v for t={t}: {encoding}")
