import torch
import torch.nn as nn
from src.multimodal_text_generation.config import config
from src.multimodal_text_generation.models.positional_embedding import PositionalEmbedding
from src.multimodal_text_generation.models.multi_head_attention import MultiHeadAttention
from src.multimodal_text_generation.models.layers import FeedForward,GELU
from src.multimodal_text_generation.models.transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self,tokenizer): 
        super().__init__()
        self.tokenizer=tokenizer
        self.pos_emb = PositionalEmbedding() 
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(6)])
        self.final_ffn = FeedForward()
        self.gelu=GELU()
        self.output_layer = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, combined_embeddings):
        x = self.pos_emb(combined_embeddings)
        x = self.blocks(x)
        x = self.final_ffn(x)
        x=self.gelu(x)
        logits=self.output_layer(x)
        return logits 