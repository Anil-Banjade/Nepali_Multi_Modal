import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import dataset,DataLoader

from src.multimodal_text_generation.config import config
from src.multimodal_text_generation.models.transformer import Transformer
from src.multimodal_text_generation.data.dataset import CaptionEmbeddingDataset, collate_fn
from src.multimodal_text_generation.utils.inference import run_inference 
from src.multimodal_text_generation.trainer import train_model 

from src.multimodal_embedding_fusion.models.model import ContrastiveModel
from src.multimodal_embedding_fusion.models.multimodal_fusion import MultiModalFusion

from torchvision import transforms


def Pipeline_test(input_image=None, input_text=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    tokenizer = AutoTokenizer.from_pretrained('NepBERTa/NepBERTa')

    contrastive_model = ContrastiveModel().to(device)
    contrastive_model.load_state_dict(torch.load('/content/drive/MyDrive/MinorProject_Nepali_MultiModal_LLM/trained_models/contrastive_model.pt'))
    contrastive_model.eval()
    
    fusion_model = MultiModalFusion().to(device)
    fusion_model.load_state_dict(torch.load('/content/drive/MyDrive/MinorProject_Nepali_MultiModal_LLM/trained_models/fused_embeddings_model.pt'))
    fusion_model.eval()
    
    transformer_model = Transformer(tokenizer).to(device)
    transformer_model.load_state_dict(torch.load('/content/drive/MyDrive/MinorProject_Nepali_MultiModal_LLM/autoregressive_model.pt'))
    transformer_model.eval()

    with torch.no_grad(): 
        if input_image is not None:
            image_features=contrastive_model.image_encoder(input_image)
        if input_text is not None:
            text_features=contrastive_model.text_encoder(
                input_ids=input_text['input_ids'],
                attention_mask=input_text['attention_mask']
            )

        if input_image is not None and input_text is not None:
            fused_embedding=fusion_model(image_features,text_features)
        elif input_image is not None:
            fused_embedding=fusion_model.image_projection(image_features)
        elif input_text is not None:
            fused_embedding=fusion_model.text_projection(text_features)
        else:
            raise ValueError('Must provide at least one input.')

        print(f"Pre-padding shape: {fused_embedding.shape}") 
        
        
        if len(fused_embedding.shape) == 3: 
            fused_embedding = fused_embedding.squeeze(1)  
        
        if fused_embedding.shape[-1] != 1024:
            if fused_embedding.shape[-1] < 1024:
                padding = torch.zeros(
                    fused_embedding.size(0),
                    1024 - fused_embedding.shape[-1]
                ).to(device)
                fused_embedding = torch.cat([fused_embedding, padding], dim=-1)
            else:   
                fused_embedding = fused_embedding[:, :1024]

        print(f"Post-padding shape: {fused_embedding.shape}")

        input_ids = torch.tensor([tokenizer.cls_token_id]).unsqueeze(0).to(device)
        
        for _ in range(config.max_seq_len - 1):
            outputs = transformer_model(fused_embedding, input_ids)
            next_token = outputs.argmax(-1)[:, -1].unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.sep_token_id:
                break

        generated_caption = tokenizer.decode(
            input_ids.squeeze().tolist(),
            skip_special_tokens=True
        )
    return generated_caption         



from PIL import Image
image_path = "/content/drive/MyDrive/MinorProject_Nepali_MultiModal_LLM/a.jpg"
raw_image = Image.open(image_path).convert("RGB")        



image_transform = transforms.Compose([  
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processed_image = image_transform(raw_image).unsqueeze(0).to(device)

tokenizer = AutoTokenizer.from_pretrained('NepBERTa/NepBERTa')
text_input = tokenizer(
    "बिरालो", 
    return_tensors='pt',
    padding='max_length',
    max_length=128,
    truncation=True
).to(device) 

# Image only
caption = Pipeline_test(input_image=processed_image)
# Text only
caption1 = Pipeline_test(input_text=text_input)        
# Multimodal
caption2 = Pipeline_test(input_image=processed_image, input_text=text_input)
print(caption)
print(caption1)
print(caption2)


# test1 = fusion_model(img1, txt1)
# test2 = fusion_model(img1, txt1)
# print("Same input diff:", (test1 - test2).abs().mean())  # Should be ~0

# # Different inputs
# test3 = fusion_model(img2, txt2)  