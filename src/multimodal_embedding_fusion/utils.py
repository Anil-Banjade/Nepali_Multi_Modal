import torch 
import pandas as pd
import numpy as np
import albumentations as A
from src.multimodal_embedding_fusion.config import Configuration

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_transforms(mode='train'):
    return A.Compose(
        [
            A.Resize(Configuration.size,Configuration.size,always_apply=True),
            A.Normalize(max_pixel_value=255.0,always_apply=True),
        ] 
    )
    
def make_train_valid_dfs(): 
    dataframe = pd.read_csv("captions.csv")  
    max_id = dataframe["id"].max() + 1 if not Configuration.debug else 100
    image_ids = np.arange(0, max_id) 
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    ) 
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe

     

     
 

