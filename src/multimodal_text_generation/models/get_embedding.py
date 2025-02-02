import torch
from transformers import AutoTokenizer, AutoModel
from src.multimodal_text_generation.config import config

from src.multimodal_text_generation.config import config


# def get_embeddings(text):
#   tokens = tokenizer(text, return_tensors="pt", padding=True, max_length=128,truncation=True)
#   with torch.no_grad():
#     outputs = text_model(**tokens)
#     embeddings = outputs.last_hidden_state.squeeze(0)
#   return embeddings

# print(tokenizer.vocab_size)
# print(len(tokenizer))


class NepBERTaEmbeddings:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('NepBERTa/NepBERTa')
        self.model = AutoModel.from_pretrained("NepBERTa/NepBERTa", from_tf=True)
        self.model.eval() 
    
    def get_token_embeddings(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        return outputs.last_hidden_state