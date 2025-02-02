import torch
from torch.utils.data import Dataset, DataLoader

class CaptionEmbeddingDataset(Dataset):
  def __init__(self,loaded_results):
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


# def collate_fn(batch):
#   captions,embeddings=zip(*batch)
#   if not all(isinstance(emb, torch.Tensor) for emb in embeddings):
#     raise ValueError("All embeddings must be tensors.")
#   max_len=max(emb.shape[0] for emb in embeddings)
#   padded_embeddings=[]
#   for emb in embeddings:
#     padding_len=max_len-emb.shape[0]
#     padded_emb=torch.nn.functional.pad(emb, (0, 0, 0, padding_len))
#     padded_embeddings.append(padded_emb)
#   padded_embeddings = torch.stack(padded_embeddings)
#   return captions,padded_embeddings


def collate_fn(batch):
    fused_embs = [item['fused_embedding'] for item in batch]
    captions = [item['caption'] for item in batch]
    
    # Process targets
    tokenized = tokenizer(captions, padding=True, return_tensors='pt')
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    
    # Pad fused embeddings to match sequence length
    max_len = input_ids.size(1)
    fused_padded = torch.stack([
        torch.cat([fe, torch.zeros(max_len - fe.size(0), config.emb_dim)])
        for fe in fused_embs
    ])
    
    return {
        'fused_embeddings': fused_padded,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }