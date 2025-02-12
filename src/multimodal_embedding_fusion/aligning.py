import torch
import pandas as pd
from tqdm import tqdm
from src.multimodal_embedding_fusion.models.model import ContrastiveModel
from src.multimodal_embedding_fusion.models.multimodal_fusion import MultiModalFusion
from src.multimodal_embedding_fusion.data.dataset import build_loaders
from src.multimodal_embedding_fusion.config import Configuration

def generate_aligned_embeddings():
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Contrastive Model
    contrastive_model = ContrastiveModel().to(device)
    contrastive_model.load_state_dict(
        torch.load('contrastive_model.pt', map_location=device, weights_only=True)
    )
    contrastive_model.eval()
    
    # 2. Load Fusion Model
    fusion_model = MultiModalFusion().to(device)
    fusion_model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    fusion_model.eval()
    
    # 3. Prepare Data Loader (same as training)
    df = pd.read_csv(Configuration.captions_path)
    tokenizer = AutoTokenizer.from_pretrained(Configuration.text_tokenizer)
    dataloader = build_loaders(df, tokenizer, mode='valid')  # Use same order as CSV
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k != 'caption'}
            captions = batch['caption']  # Original captions from dataset
            
            # Get features from contrastive model
            image_features = contrastive_model.image_encoder(batch['image'])
            text_features = contrastive_model.text_encoder(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Get fused embeddings
            fused_embeddings = fusion_model(image_features, text_features)
            
            # Store with captions
            for i in range(fused_embeddings.shape[0]):
                results.append({
                    'caption': captions[i],
                    'embedding': fused_embeddings[i].cpu().numpy()
                })
    
    # Convert to DataFrame and verify
    result_df = pd.DataFrame(results)
    print(f"Generated {len(result_df)} embeddings")
    print("Sample entry:")
    print(result_df.iloc[0]['caption'])
    print("Embedding shape:", result_df.iloc[0]['embedding'].shape)
    
    return result_df

# Execute and save
if __name__ == "__main__":
    embedding_df = generate_aligned_embeddings()
    
    # Save with captions
    torch.save({
        'captions': embedding_df['caption'].values,
        'embeddings': np.stack(embedding_df['embedding'].values),
        'metadata': {
            'num_samples': len(embedding_df),
            'embedding_dim': embedding_df['embedding'][0].shape[0],
            'generation_date': pd.Timestamp.now().isoformat()
        }
    }, 'aligned_caption_embeddings.pt')