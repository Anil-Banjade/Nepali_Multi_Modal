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
      captions, embeddings = batch
      embeddings=embeddings.to(device) 

      tokens=model.tokenizer(captions,return_tensors='pt',padding=True,max_length=128,truncation=True)
      target_tokens=tokens['input_ids'].to(device)

      outputs=model(embeddings) 
 
      outputs=outputs[:,1:-1,:]
      targets=target_tokens[:,1:] 


      outputs = outputs.contiguous().view(-1, config.vocab_size)
      targets = targets.contiguous().view(-1)

      loss=criterion(outputs,targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss+=loss.item()

      if batch_idx%1000==0:
        print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_loss=total_loss/len(dataloader)
    print(f'Epoch [{epoch}/{num_epochs}], Average Loss: {avg_loss:.4f}')
