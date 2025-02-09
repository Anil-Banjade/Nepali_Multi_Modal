import torch
import torch.nn as nn
from src.multimodal_text_generation.config import config
from src.multimodal_text_generation.models.positional_embedding import PositionalEmbedding
from src.multimodal_text_generation.models.multi_head_attention import MultiHeadAttention
from src.multimodal_text_generation.models.layers import FeedForward
from src.multimodal_text_generation.models.layers import GELU
from src.multimodal_text_generation.models.transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self,tokenizer): 
        super().__init__()
        self.tokenizer=tokenizer
        self.pos_emb = PositionalEmbedding()  
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(6)])
        
        self.fused_proj = nn.Linear(768, config.emb_dim)  
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        
        self.gelu=GELU()
        self.output_layer = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, fused_emb, token_ids=None): 
        fused_emb = self.fused_proj(fused_emb).unsqueeze(1)
        if token_ids is not None:
            token_embs = self.token_embedding(token_ids)  
            combined_embeddings = torch.cat([fused_emb, token_embs], dim=1)
        else:
            combined_embeddings = fused_emb
        
        x = self.pos_emb(combined_embeddings)
        x = self.blocks(x)
        logits=self.output_layer(x)
        return logits  