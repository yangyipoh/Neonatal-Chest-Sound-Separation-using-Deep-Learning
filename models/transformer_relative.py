import torch
from torch import nn, Tensor

from models.common import FeedForwardModule, RelativeMultiheadSelfAttention


class TransformerRelativeBlock(nn.Module):
    def __init__(self, 
            encoder_dim:int=512,
            ffn_expansion:int=4,
            num_attention_heads:int=4,
            dropout:float=0.1,
        ):
        super().__init__()
        self.self_attention = RelativeMultiheadSelfAttention(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=dropout,
        )
        self.feedforward = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=ffn_expansion,
            dropout_p=dropout,
        )
        self.attn_norm = nn.LayerNorm(encoder_dim)
        self.ffn_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.attn_norm(x)
        x = self.self_attention(x) + x
        x = self.ffn_norm(x)
        x = self.feedforward(x) + x
        return x


class TransformerRelativeEncoder(nn.Module):
    def __init__(self,
            encoder_dim:int=512,
            num_layers:int=4,
            ffn_expansion:int=4,
            num_attention_heads:int=4,
            dropout:float=0.1,
        ):
        super().__init__()
        self.layers = nn.ModuleList([TransformerRelativeBlock(
            encoder_dim = encoder_dim,
            ffn_expansion = ffn_expansion,
            num_attention_heads = num_attention_heads,
            dropout = dropout,
        ) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x
