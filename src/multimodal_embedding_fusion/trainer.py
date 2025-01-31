import torch
import itertools
from tqdm.autonotebook import tqdm
from utils import get_lr
from config import Configuration
from torch import nn

def train_epoch(model,train_loader,optimizer,lr_scheduler,step):
    total_loss=0
    num_samples=0
    model.train() 
    progress_bar=tqdm(train_loader,total=len(train_loader))
    for batch in progress_bar:
        batch={k:v.to(Configuration.device) for k,v in batch.items() if k!='caption'}
        loss=model(batch)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        batch_size = batch["image"].size(0)
        total_loss+=loss.item()*batch_size
        num_samples+=batch_size

        current_loss=total_loss/num_samples
        progress_bar.set_postfix(loss=f'{current_loss:.4f}', 
                               lr=f'{optimizer.param_groups[0]["lr"]:.6f}')
    return total_loss/num_samples
  
def valid_epoch(model, valid_loader):
    total_loss = 0
    num_samples = 0
    
    model.eval()
    progress_bar = tqdm(valid_loader, desc='Validation')
    
    for batch in progress_bar:
        batch = {k: v.to(Configuration.device) for k, v in batch.items() if k != "caption"}
        
        with torch.no_grad():
            loss = model(batch)
        
        batch_size = batch["image"].size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size
        
        current_loss = total_loss / num_samples
        progress_bar.set_postfix(loss=f'{current_loss:.4f}')
    
    return total_loss / num_samples
    
    
    
      
def train(train_loader,valid_loader,model):
    optimizer=torch.optim.AdamW([
        {'params':model.image_encoder.parameters(),'lr':Configuration.image_encoder_lr},
        {'params':model.text_encoder.parameters(),'lr':Configuration.text_encoder_lr},
        {'params':list(model.image_projection.parameters())+list(model.text_projection.parameters()),
         'lr':Configuration.head_lr,
         'weight_decay':Configuration.weight_decay}
    ])
    
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=Configuration.patience,
        factor=Configuration.factor
    )
    best_loss=float('inf')
    for epoch in range(Configuration.epochs):
        print(f'\nEpoch: {epoch+1}/{Configuration.epochs}')
        train_loss=train_epoch(model,train_loader,optimizer,lr_scheduler,'epoch')
        print(f'Training Loss: {train_loss:.4f}')
        valid_loss=valid_epoch(model,valid_loader)
        print(f'Validation Loss: {valid_loss:.4f}')
        
        if valid_loss<best_loss:
            best_loss=valid_loss
            torch.save(model.state_dict(),'best.pt')
            print(f'Saved best model. Loss: {best_loss:.4f}')
        lr_scheduler.step(valid_loss)