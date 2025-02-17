import torch
class Configuration:
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_tokenizer='NepBERTa/NepBERTa'
    max_length=128
    text_encoder='NepBERTa/NepBERTa'
    text_embedding=768
    image_encoder='resnet50'
    image_embedding=2048
    pretrained=True
    trainable=True
    debug=False
    batch_size=32
    num_workers=0
    head_lr = 1e-3
    text_encoder_lr=1e-5
    image_encoder_lr=1e-5
    patience=1
    factor=0.8
    epochs=3
    temperature=0.9
    weight_decay = 1e-3
    size=224
    num_projection_layers=1
    projection_dim=512 
    fusion_dim=512  
    dropout=0.3
    num_epochs=3

    # image_path = "data/ficker8k_images/ficker8k_images"
    # captions_path = "data/captions.csv"
    
    
    
    