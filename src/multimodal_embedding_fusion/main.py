import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer
from src.multimodal_embedding_fusion.config import Configuration
from src.multimodal_embedding_fusion.utils import make_train_valid_dfs
from src.multimodal_embedding_fusion.data.dataset import build_loaders
from src.multimodal_embedding_fusion.models.model import ContrastiveModel
from src.multimodal_embedding_fusion.models.multimodal_fusion import train_combined
from src.multimodal_embedding_fusion.trainer import train

def setup_data():
    """Setup and preprocess the dataset"""
    df = pd.read_csv(
        "/content/drive/MyDrive/MinorProject_Nepali_MultiModal_LLM/translated_nepali_captions.txt",
        delimiter='#',
        names=['image', 'caption'],
        engine='python',

    ) 
    df['caption'] = df['caption'].apply(lambda x: re.sub(r'^\d+\s+', '', x))
    df['id'] = [i // 5 for i in range(len(df))]

    print("\nSample validation:")

    for i in range(0, 50, 5):
        print(f"Image: {df.iloc[i]['image']}")
        print(f"Caption 1: {df.iloc[i]['caption']}")
        print(f"Caption 2: {df.iloc[i+1]['caption']}\n")

    df.to_csv("captions.csv", index=False)  

    Configuration.image_path = "/content/drive/MyDrive/MinorProject_Nepali_MultiModal_LLM/trained_models/datasets/ficker8k_images"
    Configuration.captions_path = "/content/drive/MyDrive/MinorProject_Nepali_MultiModal_LLM/captions.csv"
     
    return df

 
def init_model_and_loaders():
    """Initialize model and data loaders"""
    model = ContrastiveModel().to(Configuration.device)
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = AutoTokenizer.from_pretrained(Configuration.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    return model, train_loader, valid_loader


def main():
    df = setup_data() 
    
    model, train_loader, valid_loader = init_model_and_loaders()
    
    train(train_loader, valid_loader, model)
    
    model_path = '/content/drive/MyDrive/Minor_project_try/contrastive_model.pt'  
    train_combined(model_path)   

# if __name__ == "__main__":
#     main()

    