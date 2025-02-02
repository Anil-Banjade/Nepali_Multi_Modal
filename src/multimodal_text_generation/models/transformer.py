import torch
import torch.nn as nn
from src.multimodal_text_generation.config import config

from src.multimodal_text_generation.models.get_embedding import NepBERTaEmbeddings

from src.multimodal_text_generation.models.positional_embedding import PositionalEmbedding
from src.multimodal_text_generation.models.multi_head_attention import MultiHeadAttention
from src.multimodal_text_generation.models.layers import FeedForward
from src.multimodal_text_generation.models.layers import GELU
from src.multimodal_text_generation.models.transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self,tokenizer): 
        super().__init__()
        self.bert_emb = NepBERTaEmbeddings() 
        self.fused_proj = nn.Linear(512, config.emb_dim)
        
        self.tokenizer=tokenizer
        self.pos_emb = PositionalEmbedding() 
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(6)])
        self.final_ffn = FeedForward()
        self.gelu=GELU()
        self.output_layer = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, fused_embeddings,input_ids):
        token_embeds = self.bert_emb.get_token_embeddings(input_ids) 
        fused_projected=self.fused_proj(fused_embeddings) 
        combined=torch.cat([
            fused_projected.unsqueeze(1), 
            token_embeds
        ],dim=1)

        x = self.pos_emb(combined)
        x = self.blocks(x)
        x = self.final_ffn(x)
        x=self.gelu(x)
        logits=self.output_layer(x) 
        return logits         