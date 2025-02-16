import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.multimodal_text_generation.config import config
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

def train_model(model,dataloader,valid_loader,num_epochs,device):
  model=model.to(device)
  optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
  criterion=nn.CrossEntropyLoss(ignore_index=0)

  best_val_loss = float('inf')
  for epoch in range(num_epochs):
    model.train()   
    total_loss=0    

    for batch_idx,batch in enumerate(dataloader):
      captions, embeddings = batch
      fused_emb=embeddings.to(device)  
      
      tokens=model.tokenizer(captions,return_tensors='pt',padding=True,max_length=128,truncation=True)
      target_ids=tokens['input_ids'].to(device) 

      outputs=model(fused_emb,target_ids[:,:-1]) 
      outputs = outputs[:, 1:, :]
 
      loss = criterion(outputs.reshape(-1, config.vocab_size), target_ids[:, 1:].contiguous().view(-1))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss+=loss.item() 

      if batch_idx%1000==0:
        print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_loss=total_loss/len(dataloader)
    print(f'Epoch [{epoch}/{num_epochs}], Average Loss: {avg_loss:.4f}')


    # Validation phase
    model.eval()
    val_loss = 0
    all_hypotheses = []
    all_references = []
    
    with torch.no_grad():
        for val_batch in valid_loader:
            val_captions, val_embeddings = val_batch
            val_fused_emb = val_embeddings.to(device)
            
            generated_ids = model.generate(val_fused_emb, max_length=128, num_beams=1, early_stopping=True)
            generated_captions = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            tokens = model.tokenizer(val_captions, return_tensors='pt', 
                                    padding=True, max_length=128, truncation=True)
            target_ids = tokens['input_ids'].to(device)
            actual_captions = model.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
            
            # Store for metrics
            all_hypotheses.extend(generated_captions)
            all_references.extend([[ref.split()] for ref in actual_captions])
            
            outputs = model(val_fused_emb, generated_ids[:, :-1])
            loss = criterion(outputs.reshape(-1, config.vocab_size), 
                            generated_ids[:, 1:].contiguous().view(-1))
            val_loss += loss.item()

        bleu_score = corpus_bleu(all_references, [h.split() for h in all_hypotheses])
        rouge = Rouge()
        rouge_scores = rouge.get_scores(all_hypotheses, [ref[0] for ref in all_references], avg=True)

        avg_val_loss = val_loss / len(valid_loader)
        print(f'Val Loss: {avg_val_loss:.4f}') 

