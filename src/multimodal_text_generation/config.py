import torch
class config:
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_tokenizer='NepBERTa/NepBERTa'
    text_encoder='NepBERTa/NepBERTa'
    d_in=768
    d_out=768
    qkv_bias=False 
    context_length=128 
    temperature=1.0
    emb_dim = 768
    num_heads = 8
    num_layers = 6
    vocab_size = 30522
    dropout = 0.1
    
  