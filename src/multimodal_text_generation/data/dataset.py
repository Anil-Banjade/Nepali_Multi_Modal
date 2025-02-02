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
        fused_embs = [item['fused_embedding'] for item in batch]
        captions = [item['caption'] for item in batch]
        
        tokenized = self.tokenizer(
            captions,
            padding='max_length',  
            max_length=min(self.tokenizer.model_max_length,512),
            return_tensors='pt',
            truncation=True,
            add_special_tokens=False
        )
        input_ids=tokenized['input_ids']
        shifted_ids=input_ids[:,:-1]
        targets=input_ids
        fused_embs_tensor=torch.stack(fused_embs)
        
        return { 
            'fused_embeddings': fused_embs_tensor,
            'shifted_input_ids':shifted_input_ids,
            'targets':targets
        }
        
    
  


