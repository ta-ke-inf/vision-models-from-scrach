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


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, N, D]
        """
        q = self.w_q(x) # [B, N, D] -> [B, N, D]
        k = self.w_k(x) # [B, N, D] -> [B, N, D]
        v = self.w_v(x) # [B, N, D] -> [B, N, D]

        batch_size, num_patch, _ = x.size()
        q_head = q.view(batch_size, num_patch, self.head, self.head_dim) # [B, N, D] -> [B, N, h, D/h]
        k_head = k.view(batch_size, num_patch, self.head, self.head_dim)
        v_head = v.view(batch_size, num_patch, self.head, self.head_dim)
