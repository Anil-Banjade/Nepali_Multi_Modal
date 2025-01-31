import torch
from torch import nn
import timm
from transformers import AutoModel,AutoConfig
from .config import Configuration 
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self,model_name=Configuration.text_encoder,pretrained=Configuration.pretrained,trainable=Configuration.trainable):
        super().__init__()
        self.model=AutoModel.from_pretrained(model_name,from_tf=True)
        for p in self.model.parameters():
            p.requires_grad=trainable
        self.target_token_idx = 0
    def forward(self,input_ids,attention_mask):
        output=self.model(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state=output.last_hidden_state
        return last_hidden_state[:,self.target_token_idx,:]
    
class ImageEncoder(nn.Module):
    def __init__(self,model_name=Configuration.image_encoder,pretrained=Configuration.pretrained,trainable=Configuration.trainable):
        super().__init__()
        try:
            self.model=timm.create_model(
                model_name,pretrained,num_classes=0,global_pool='avg'
            )
        except:
            print(f"Error loading model {model_name}. Available models:")
            print(timm.list_models("*resnet*")) 
            raise
        for p in self.model.parameters():
            p.requires_grad=trainable
        
    
    def forward(self,x):
        return self.model(x) 
    
    
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=Configuration.projection_dim,
        dropout=Configuration.dropout
    ):
        super().__init__()
        self.projection=nn.Linear(embedding_dim,projection_dim)
        self.gelu=nn.GELU()
        self.fc=nn.Linear(projection_dim,projection_dim)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(projection_dim)
        
    def forward(self,x):
        projected=self.projection(x)
        x=self.gelu(projected)
        x=self.fc(x)
        x=self.dropout(x)
        x=x+projected
        x=self.layer_norm(x)
        return x


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
    
def calc_similarity_and_labels(image_embeddings,text_embeddings,temperature):
    logits=torch.matmul(image_embeddings, text_embeddings.T) / temperature
    texts_similarity=torch.matmul(text_embeddings,text_embeddings.T)/temperature
    images_similarity=torch.matmul(image_embeddings,image_embeddings.T)
    combined_similarity=(texts_similarity+images_similarity)/(2*temperature)
    labels=F.softmax(combined_similarity,dim=-1)
    return logits,labels 
     
class ContrastiveModel(nn.Module):
    def __init__( 
        self,
        temperature=Configuration.temperature,
        image_embedding=Configuration.image_embedding,
        text_embedding=Configuration.text_embedding
    ): 
        super().__init__()
        self.image_encoder=ImageEncoder()
        self.text_encoder=TextEncoder()
        self.image_projection=ProjectionHead(embedding_dim=image_embedding)
        self.text_projection=ProjectionHead(embedding_dim=text_embedding)
        self.temperature=temperature
        
    def forward(self,batch): 
        image_features=self.image_encoder(batch['image'])
        text_features=self.text_encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        image_embeddings=self.image_projection(image_features)
        text_embeddings=self.text_projection(text_features)
        
        logits,labels=calc_similarity_and_labels(
            image_embeddings,text_embeddings,self.temperature
        )
        text_loss=cross_entropy(logits,labels,reduction='none')
        image_loss=cross_entropy(logits.T,labels.T,reduction='none')
        loss=(image_loss+text_loss)/2.0
        return loss.mean() 
        
        