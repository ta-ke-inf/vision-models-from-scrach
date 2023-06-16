import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.vit_input_layer import VitInputLayer
from modules.encoder_block import EncoderBlock

class VisionTransformer(nn.Module):
    def __init__(self,
                 in_channels:int = 3,
                 image_size:int = 32,
                 emb_dim:int = 384,
                 num_patch_row:int = 2,
                 hidden_dim:int = 384*4,
                 head:int = 8,
                 dropout:float = 0.,
                 num_blocks:int = 7,
                 num_classes:int = 10
                 ) -> None:
        super(VisionTransformer, self).__init__()

        self.vit_input_layer = VitInputLayer(
            in_channels = in_channels,
            image_size = image_size,
            emb_dim = emb_dim,
            num_patch_row = num_patch_row
        )

        self.encoder = nn.Sequential(
            *[EncoderBlock(hidden_dim = hidden_dim,
                             emb_dim = emb_dim,
                             head = head,
                             dropout = dropout)
                for _ in range(num_blocks)]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input image

        Returns:
            torch.Tensor: prediction [B, num_classes]
        """
        x = self.vit_input_layer(x)
        x = self.encoder(x)
        cls_token = x[:, 0] # [B, N, D] -> [B, D]
        prediction = self.mlp_head(cls_token) # [B, D] -> [B, num_classes]

        return prediction

if __name__ == "__main__":
    x = torch.randn([16, 3, 32, 32])
    vision_transformer = VisionTransformer(in_channels=3, image_size=32)
    x = vision_transformer(x)
    print(x.shape)
