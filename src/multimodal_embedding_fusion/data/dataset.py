import cv2
import torch
import torch.nn as nn
from src.multimodal_embedding_fusion.config import Configuration
from src.multimodal_embedding_fusion.utils import get_transforms


class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self,image_filenames,captions,tokenizer,transforms):
        self.image_filenames=image_filenames
        self.captions=list(captions)
        self.encoded_captions=tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=Configuration.max_length
        )
        self.transforms=transforms
    
    def __getitem__(self,idx):
        item={
            key:torch.tensor(values[idx])
            for key,values in self.encoded_captions.items()
        }
        image=cv2.imread(f"{Configuration.image_path}/{self.image_filenames[idx]}")
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=self.transforms(image=image)['image']
        item['image']=torch.tensor(image).permute(2,0,1).float()
        item['caption']=self.captions[idx]
        return item
    def __len__(self):
        return len(self.captions)
    
def build_loaders(dataframe,tokenizer,mode):
    transforms=get_transforms(mode=mode)
    dataset=ImageTextDataset(
        dataframe['image'].values,
        dataframe['caption'].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader=torch.utils.data.DataLoader(
        dataset,
        batch_size=Configuration.batch_size,
        num_workers=Configuration.num_workers,
        shuffle=True if mode=='train' else False,
    )
    return dataloader


