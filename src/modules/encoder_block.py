import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_self_attention import MultiHeadSelfAttention

class EncoderBlock(nn.Module):
    def __init__(self,
                 hidden_dim = 384*4,
                 emb_dim:int = 384,
                 head:int = 3,
                 dropout:float = 0.
                 ) -> None:

        super(EncoderBlock, self).__init__()
        self.emb_dim = emb_dim
        self.head = head
        self.dropout = dropout

        self.layer_norm_1 = nn.LayerNorm(emb_dim)
        self.mhsa = MultiHeadSelfAttention(emb_dim=self.emb_dim, head=self.head, dropout=self.dropout)

        self.layer_norm_2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, N, D]

        Returns:
            torch.Tensor: [B, N, D]
        """
        x = self.mhsa(self.layer_norm_1(x)) + x
        x = self.mlp(self.layer_norm_2(x)) + x
        return x

if __name__ == "__main__":
    x = torch.randn([16, 5, 384])
    encoder_block = EncoderBlock(hidden_dim=384*4, emb_dim=384, head=3)
    x = encoder_block(x)
    print(x.shape)
