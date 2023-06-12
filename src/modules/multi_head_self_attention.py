import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 emb_dim:int = 384,
                 head:int = 3,
                 dropout:float = 0.
                 ) -> None:
        """
        Args:
            emb_dim (int, optional): Vector length after embedded. Defaults to 384.
            head (int, optional): head num. Defaults to 3.
            dropout (float, optional): drop out ratio. Defaults to 0..
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head # h
        self.emb_dim = 384 # D
        self.dropout = dropout
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim ** 0.5

        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, N, D]

        Returns:
            torch.Tensor: output tensor, [B, N, D]
        """
        q = self.w_q(x) # [B, N, D] -> [B, N, D]
        k = self.w_k(x) # [B, N, D] -> [B, N, D]
        v = self.w_v(x) # [B, N, D] -> [B, N, D]

        batch_size, num_patch, _ = x.size()

        q_heads = q.view(batch_size, num_patch, self.head, self.head_dim) # [B, N, D] -> [B, N, h, D/h]
        k_heads = k.view(batch_size, num_patch, self.head, self.head_dim)
        v_heads = v.view(batch_size, num_patch, self.head, self.head_dim)

        q_heads = torch.permute(q_heads, (0, 2, 1, 3)) # [B, N, h, D/h] -> [B, h, N, D/h]
        k_heads = torch.permute(k_heads, (0, 2, 1, 3)) # [B, N, h, D/h] -> [B, h, N, D/h]
        v_heads = torch.permute(v_heads, (0, 2, 1, 3)) # [B, N, h, D/h] -> [B, h, N, D/h]

        # Layer Normalization
        k_heads_t = torch.permute(k_heads, (0, 1, 3, 2)) # [B, h, D/h, N]
        attn_weight = F.softmax((q_heads @ k_heads_t) / self.sqrt_dh, dim=2) # [B, h, N, N]

        x = attn_weight @ v_heads # [B, h, N, D/h]
        x = torch.permute(x, (0, 2, 1, 3)).reshape(batch_size, num_patch, self.emb_dim)

        return x

if __name__ == "__main__":
    mhsa = MultiHeadSelfAttention()
    x = torch.randn([16, 5, 384])
    x = mhsa.forward(x)
    print(x.shape)
