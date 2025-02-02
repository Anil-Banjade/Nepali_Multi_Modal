import torch
from torch.utils.data import Dataset, DataLoader
from src.multimodal_text_generation.config import config

class CaptionEmbeddingDataset(Dataset):
  def __init__(self,loaded_results,tokenizer):
    self.tokenizer=tokenizer
    self.captions=[item[0] for item in loaded_results]
    self.fused_embeddings=[item[1] for item in loaded_results]
    # for i, (caption, embedding) in enumerate(zip(self.captions, self.embeddings)):
    #   print(f"Index {i}: Caption type: {type(caption)}, Embedding type: {type(embedding)}")
  def __len__(self):
    return len(self.captions)
  
  def __getitem__(self,idx):  
    return {
      'fused_embedding':self.fused_embeddings[idx],
      'caption':self.captions[idx]
    }

  def collate_fn(self, batch):
        fused_embs = [item['fused_embedding'] for item in batch]
        captions = [item['caption'] for item in batch]
        
        # Tokenize without truncation first to get actual lengths
        tokenized = self.tokenizer(
            captions,
            padding='longest',  # First get natural lengths
            return_tensors='pt',
            truncation=False
        )
        
        # Find max length considering both text and fused embeddings
        max_text_len = tokenized['input_ids'].size(1)
        max_fused_len = max(fe.size(0) for fe in fused_embs)
        max_len = max(max_text_len, max_fused_len)
        
        # Now apply truncation if needed
        if max_len > self.tokenizer.model_max_length:
            tokenized = self.tokenizer(
                captions,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt'
            )
            max_len = self.tokenizer.model_max_length
        
        # Pad fused embeddings
        fused_padded = torch.stack([
            torch.cat([
                fe[:max_len],  # Truncate if needed
                torch.zeros(max(0, max_len - fe.size(0)), config.emb_dim)
            ]) for fe in fused_embs
        ])
        
        # Pad text sequences to match fused length
        if tokenized['input_ids'].size(1) < max_len:
            pad_amount = max_len - tokenized['input_ids'].size(1)
            tokenized['input_ids'] = torch.nn.functional.pad(
                tokenized['input_ids'], 
                (0, pad_amount), 
                value=self.tokenizer.pad_token_id
            )
            tokenized['attention_mask'] = torch.nn.functional.pad(
                tokenized['attention_mask'],
                (0, pad_amount),
                value=0
            )
        
        return {
            'fused_embeddings': fused_padded,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
    
  # def collate_fn(self, batch):
  #   fused_embs = [item['fused_embedding'] for item in batch]
  #   captions = [item['caption'] for item in batch]
    
  #   # Process targets
  #   tokenized = self.tokenizer(captions, padding=True, return_tensors='pt')
  
    
  #   # Pad fused embeddings to match sequence length
  #   max_len = tokenized['input_ids'].size(1)
  #   fused_padded = torch.stack([
  #       torch.cat([fe, torch.zeros(max_len - fe.size(0), config.emb_dim)])
  #       for fe in fused_embs
  #   ])
    
  #   return {
  #           'fused_embeddings': fused_padded,
  #           'input_ids': tokenized['input_ids'],
  #           'attention_mask': tokenized['attention_mask']
  #   }



