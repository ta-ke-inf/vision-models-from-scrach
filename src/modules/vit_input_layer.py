import torch
import torch.nn as nn


class VitInputLayer(nn.Module):
    def __init__(self,
        in_channels: int = 3,
        image_size: int = 32,
        emb_dim: int = 384,
        num_patch_row: int = 2
    ) -> None:
        super(VitInputLayer, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.num_patch = self.num_patch_row ** 2
        self.patch_size = self.image_size // self.num_patch_row

        self.patch_emb_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        self.class_token = nn.Parameter(1, 1, self.emb_dim)

        self.pos_emb = nn.Parameter(1, self.num_patch + 1, self.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
