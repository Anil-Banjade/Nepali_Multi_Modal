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
        
        # self.fused_proj = nn.Linear(1024, config.emb_dim)  
        self.fused_proj=nn.Identity()
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

    def generate(self, fused_emb, max_length=128, num_beams=5, early_stopping=True):
        batch_size = fused_emb.size(0)
        device = fused_emb.device
        
        input_ids = torch.full((batch_size, 1), 
                             self.tokenizer.cls_token_id, 
                             dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            outputs = self(fused_emb, input_ids)
            next_token_logits = outputs[:, -1, :]
            
            if num_beams > 1:
                next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                next_token_scores = next_token_scores.view(batch_size, -1)
                
                scores, indices = torch.topk(
                    next_token_scores, 
                    num_beams,
                    dim=1,
                    largest=True,
                    sorted=True
                )
                
                input_ids = torch.cat([
                    input_ids.repeat_interleave(num_beams, dim=0),
                    indices.view(-1, 1)
                ], dim=1)
                
                fused_emb = fused_emb.repeat_interleave(num_beams, dim=0)
            else:
                
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

            if (input_ids[:,-1] == self.tokenizer.sep_token_id).all() and early_stopping:
                break

        return input_ids