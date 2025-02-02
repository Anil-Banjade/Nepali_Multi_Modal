import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.multimodal_text_generation.config import config
from tqdm import tqdm


def train_model(model,dataloader,num_epochs,device):
  model=model.to(device)
  optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
  criterion=nn.CrossEntropyLoss(ignore_index=0)

  for epoch in range(num_epochs):
    model.train()
    total_loss=0

    for batch_idx,batch in enumerate(dataloader):
      fused_embs = batch['fused_embeddings'].to(device)
      shifted_input_ids=batch['shifted_input_ids'].to(device)
      targets = batch['targets'].to(device)
      
      logits=model(fused_embs,shifted_input_ids)
      

      loss = criterion(logits.view(-1, config.vocab_size), targets.view(-1))
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss+=loss.item()

      if batch_idx%100==0:
        print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_loss=total_loss/len(dataloader)
    print(f'Epoch [{epoch}/{num_epochs}], Average Loss: {avg_loss:.4f}')
