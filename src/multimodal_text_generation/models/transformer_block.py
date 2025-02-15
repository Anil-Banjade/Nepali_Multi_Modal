import torch
import torch.nn as nn
from src.multimodal_text_generation.config import config
from src.multimodal_text_generation.models.layers import LayerNorm
from src.multimodal_text_generation.models.multi_head_attention import MultiHeadAttention
from src.multimodal_text_generation.models.layers import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = LayerNorm(config.emb_dim)
        self.attention = MultiHeadAttention( 
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            dropout=config.dropout,
            num_heads=config.num_heads
        )
        self.ln2 = LayerNorm(config.emb_dim)
        self.ffn = FeedForward()

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x 