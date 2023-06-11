import torch
import torch.nn as nn


class VitInputLayer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        emb_dim: int = 384,
        num_patch_row: int = 2,
    ) -> None:
        super(VitInputLayer, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.num_patch = self.num_patch_row**2
        self.patch_size = self.image_size // self.num_patch_row

        self.patch_emb_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.class_token = nn.Parameter(
            torch.randn(1, 1, self.emb_dim)
            )

        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch + 1, self.emb_dim )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input image tensor, [N,C,H,W]

        Returns:
            torch.Tensor: output tensor, [B,N,D]
                D is length of embedding: 384
        """
        x = self.patch_emb_layer(x)  # [N,C,H,W]->[N,D,H/P,W/P]
        x = x.flatten(2)  # [N,D,H/P,W/P]->[N,D,H/P*W/P]
        x = x.transpose(1, 2) # [N,D,H/P*W/P] -> [N,H/P*W/P,D]
        class_token = self.class_token.repeat([x.size(0), 1, 1]) # [1,1,D]->[N,1,D]
        z_0 = torch.cat((x, class_token), dim=1) # [N,Np+1,D]

        z_0 = z_0 + self.pos_emb
        return z_0

if __name__ == "__main__":

    in_channels: int = 3
    image_size: int = 32
    num_batch: int = 16
    imgs = torch.randn(num_batch, in_channels, image_size, image_size)

    vit_input_layer = VitInputLayer()
    x = vit_input_layer.forward(imgs)
    print(x.shape)
