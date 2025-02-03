import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from src.multimodal_text_generation.models.transformer import Transformer
def generate_caption(model, tokenizer, fused_embedding, device, max_length=50):
    model.eval()
    with torch.no_grad(): 
        try:
            fused_embedding = fused_embedding.view(1, 1, -1).to(device)
            print(f"Image embedding shape: {fused_embedding.shape}")
            
            tokens = tokenizer("[CLS]", return_tensors="pt", padding=True, max_length=128, truncation=True)
            with torch.no_grad():
                bos_output = text_model(**tokens)
                bos_embedding = bos_output.last_hidden_state[:, 0].unsqueeze(1)  
                bos_embedding = bos_embedding.to(device)
            
            print(f"BOS embedding shape: {bos_embedding.shape}")
            
            current_embeddings = torch.cat([fused_embedding, bos_embedding], dim=1)
            print(f"Current embeddings shape: {current_embeddings.shape}")
           
            generated_tokens = [tokenizer.cls_token_id]
 
            print("Starting generation loop...")
            for i in range(max_length):
                try:
                    outputs = model(current_embeddings)
                    next_token_logits = outputs[:, -1, :] 

                    temperature = 0.9  
                    scaled_logits = next_token_logits / temperature

                    for token_id in tokenizer.all_special_ids:
                        if token_id != tokenizer.sep_token_id:
                            scaled_logits[0, token_id] = float('-inf')

                    probs = torch.softmax(scaled_logits, dim=-1)

                    k = 40
                    top_k_probs, top_k_indices = torch.topk(probs[0], k)

                    next_token = top_k_indices[torch.multinomial(top_k_probs, 1)].item()

                    print(f"Generated token {i}: {next_token} -> {tokenizer.decode([next_token])}")

                    if next_token == tokenizer.sep_token_id and len(generated_tokens) > 3:
                        print("End token generated after meaningful sequence")
                        break

                    if len(generated_tokens) >= 3 and all(t == next_token for t in generated_tokens[-3:]):
                        print("Detected repetition, skipping token")
                        continue

                    generated_tokens.append(next_token)

                    new_token_input = tokenizer.decode([next_token], skip_special_tokens=False)
                    token_output = text_model(**tokenizer(new_token_input, 
                                                        return_tensors="pt", 
                                                        padding=True, 
                                                        max_length=128,
                                                        truncation=True).to(device))
                    token_embedding = token_output.last_hidden_state[:, 0].unsqueeze(1)
                    
                    current_embeddings = torch.cat([current_embeddings, token_embedding], dim=1)
                    
                    if i % 2 == 0:
                        partial_caption = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        print(f"Partial caption: {partial_caption}")

                except Exception as e:
                    print(f"Error in generation loop at step {i}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    break

            final_caption = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"Final generated caption: {final_caption}")

            if len(final_caption.strip()) > 0:
                return final_caption.strip()
            else:
                print("Generated caption was empty, retrying...")
                return None

        except Exception as e:
            print(f"Error in generate_caption: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

def load_model(load_path, device):
    checkpoint = torch.load(load_path, map_location='cpu')
    tokenizer = AutoTokenizer.from_pretrained('NepBERTa/NepBERTa')
    model = Transformer(tokenizer)
    filtered_state_dict = {
        k: v for k, v in checkpoint['model_state_dict'].items()
        if k in model.state_dict()
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, tokenizer


def run_inference(model_path, test_image_embedding, device, max_attempts=3):
    try:
        print("Loading model...")
        model, tokenizer = load_model(model_path, device)

        for attempt in range(max_attempts):
            print(f"\nAttempt {attempt + 1}/{max_attempts}")
            generated_caption = generate_caption(
                model,
                tokenizer,
                test_image_embedding,
                device
            )

            if generated_caption:
                return generated_caption

        print(f"Failed to generate caption after {max_attempts} attempts")
        return None

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

