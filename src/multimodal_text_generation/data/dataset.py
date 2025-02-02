import torch
from torch.utils.data import Dataset, DataLoader
from src.multimodal_text_generation.config import config

class CaptionEmbeddingDataset(Dataset):
  def __init__(self,loaded_results,tokenizer):
    self.tokenizer=tokenizer
    self.captions=[item[0] for item in loaded_results]
    self.fused_embeddings=[item[1] for item in loaded_results]
  def __len__(self):
    return len(self.captions)
  
  def __getitem__(self,idx):  
    return {
      'fused_embedding':self.fused_embeddings[idx],
      'caption':self.captions[idx]
    }

  def collate_fn(self, batch): 
        captions = [item['caption'] for item in batch]
        
        tokenized = self.tokenizer(
            captions,
            padding='max_length',  
            max_length=config.max_seq_len,
            return_tensors='pt',
            truncation=True,
            add_special_tokens=True
        )
        
        fused_embs = torch.stack([item['fused_embedding'] for item in batch])
        input_ids = tokenized['input_ids']
        shifted_ids = input_ids[:, :-1] 
        targets = input_ids[:, 1:] 
        
        
        return { 
            'fused_embeddings': fused_embs,
            'shifted_input_ids':shifted_ids,
            'targets':targets
        }
        
    
  


