import torch
import torch.nn as nn
from models.model import ProjectionHead,ContrastiveModel
from config import Configuration
from tqdm.autonotebook import tqdm
from dataset import build_loaders
from transformers import AutoTokenizer
from utils import make_train_valid_dfs


class MultiModalFusion(nn.Module):
    def __init__(
        self,
        image_embedding=Configuration.image_embedding,
        text_embedding=Configuration.text_embedding, 
        fusion_dim=Configuration.fusion_dim
    ):
        super().__init__()
        self.image_projection=ProjectionHead(embedding_dim=image_embedding)
        self.text_projection=ProjectionHead(embedding_dim=text_embedding)
        self.fusion_dim=fusion_dim 
        
        self.cross_attention=nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8, 
            dropout=0.1
        ) 
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
    
    def forward(self,image_features,text_features):
        image_projection=self.image_projection(image_features)
        text_projection=self.text_projection(text_features)

        if len(image_projection.shape) == 2:
            image_projection = image_projection.unsqueeze(0)
        if len(text_projection.shape) == 2:
            text_projection = text_projection.unsqueeze(0)
        
        fused,_=self.cross_attention(
            query=image_projection,
            key=text_projection,
            value=text_projection
        )
        return self.final_fusion(fused)


# def generate_fused_embeddings(model_path='best.pt'):
#     train_df, _ = make_train_valid_dfs()
#     tokenizer = AutoTokenizer.from_pretrained(Configuration.text_tokenizer)
#     data_loader = build_loaders(train_df, tokenizer, mode="train")
    
#     contrastive_model = ContrastiveModel().to(Configuration.device)
#     contrastive_model.load_state_dict(torch.load(model_path))
#     contrastive_model.eval()

#     fusion_model = MultiModalFusion().to(Configuration.device)
#     fusion_model.eval()
    
#     all_fused_embeddings = []
#     progress_bar = tqdm(data_loader, desc='Generating fused embeddings')
    
#     with torch.no_grad():
#         for batch in progress_bar:
#             batch = {k: v.to(Configuration.device) for k, v in batch.items() if k != 'caption'}
#             image_features = contrastive_model.image_encoder(batch['image'])
#             text_features = contrastive_model.text_encoder(
#                 input_ids=batch['input_ids'],
#                 attention_mask=batch['attention_mask']
#             )
            
#             fused = fusion_model(image_features, text_features)
            
#             all_fused_embeddings.append({
#                 'fused_embedding': fused.cpu(),
#                 'caption': batch.get('caption', None)
#             })
            
#     return all_fused_embeddings


def train_combined():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = AutoTokenizer.from_pretrained(Configuration.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    contrastive_model = ContrastiveModel().to(Configuration.device)
    contrastive_model.load_state_dict(torch.load('best.pt'))
    contrastive_model.eval() 
    
    fusion_model = MultiModalFusion().to(Configuration.device)
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  
    
    best_loss = float('inf')
    
    for epoch in range(Configuration.num_epochs):
        fusion_model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            batch = {k: v.to(Configuration.device) for k, v in batch.items() if k != 'caption'}
            
            
            with torch.no_grad():
                image_features = contrastive_model.image_encoder(batch['image'])
                text_features = contrastive_model.text_encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
            
            fused = fusion_model(image_features, text_features)
            
            loss = criterion(fused, (image_features + text_features) / 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}')
        
        # Save best model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(fusion_model.state_dict(), 'best_fusion.pt')

    print('Training completed!')


 
 