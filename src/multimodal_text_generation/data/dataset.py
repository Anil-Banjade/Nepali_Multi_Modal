import torch
from torch.utils.data import Dataset, DataLoader

class CaptionEmbeddingDataset(Dataset): 
  def __init__(self,loaded_results,tokenizer):
    self.captions=loaded_results['captions']
    self.embeddings=loaded_results['embeddings']
    self.tokenizer=tokenizer
  def __len__(self):
    return len(self.captions) 
  def __getitem__(self,idx):
    return self.captions[idx],torch.tensor(self.embeddings[idx])

def collate_fn(batch):
  captions,embeddings=zip(*batch) 
  embeddings=torch.stack(embeddings)
  return captions,embeddings  


