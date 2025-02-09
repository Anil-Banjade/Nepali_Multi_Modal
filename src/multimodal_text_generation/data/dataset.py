import torch
from torch.utils.data import Dataset, DataLoader

class CaptionEmbeddingDataset(Dataset): 
  def __init__(self,loaded_results,tokenizer):
    self.captions=[item[0] for item in loaded_results]
    self.embeddings=[item[1] for item in loaded_results]
    self.tokenizer=tokenizer
  def __len__(self):
    return len(self.captions)
  def __getitem__(self,idx):
    return self.captions[idx],self.embeddings[idx]

def collate_fn(batch):
  captions,embeddings=zip(*batch)
  if not all(isinstance(emb, torch.Tensor) for emb in embeddings):
    raise ValueError("All embeddings must be tensors.")
  max_len=max(emb.shape[0] for emb in embeddings)
  padded_embeddings=[] 
  for emb in embeddings:
    padding_len=max_len-emb.shape[0]
    padded_emb=torch.nn.functional.pad(emb, (0, 0, 0, padding_len))
    padded_embeddings.append(padded_emb)
  padded_embeddings = torch.stack(padded_embeddings)
  return captions,padded_embeddings


