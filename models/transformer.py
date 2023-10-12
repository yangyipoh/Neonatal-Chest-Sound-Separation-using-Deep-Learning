import torch
from torch import nn, Tensor

from models.common import PositionalEncoding, FeedForwardModule, MultiheadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, 
            encoder_dim:int=512,
            ffn_expansion:int=4,
            num_attention_heads:int=4,
            dropout:float=0.1,
        ):
        super().__init__()
        self.self_attention = MultiheadSelfAttention(
            embed_dim=encoder_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feedforward = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=ffn_expansion,
            dropout_p=dropout,
        )
        
        self.attn_norm = nn.LayerNorm(encoder_dim)
        self.ffn_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.self_attention(x) + x
        x = self.attn_norm(x)
        x = self.feedforward(x) + x
        x = self.ffn_norm(x)
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self,
        encoder_dim:int=256,
        num_layers:int=4,
        num_attention_heads:int=4,
        ffn_expansion:int=4,
        dropout:float=0.1,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model=encoder_dim, max_len=600)
        self.layers = nn.ModuleList([TransformerBlock(
            encoder_dim=encoder_dim,
            ffn_expansion = ffn_expansion,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        ) for _ in range(num_layers)])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.size()
        x += self.pos_enc(seq_len)
        for layer in self.layers:
            x = layer(x)
        return x
