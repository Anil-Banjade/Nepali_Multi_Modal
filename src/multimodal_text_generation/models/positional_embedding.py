import torch
import torch.nn as nn
from src.multimodal_text_generation.config import config

class PositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(config.context_length, config.emb_dim)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).expand(x.size(0), seq_len)
        return x + self.emb(positions) 