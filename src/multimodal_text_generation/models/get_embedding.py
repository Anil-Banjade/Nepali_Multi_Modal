from src.multimodal_text_generation.config import config

# tokenizer=AutoTokenizer.from_pretrained(config.text_tokenizer)
# text_model = AutoModel.from_pretrained(config.text_encoder,from_tf=True)

def get_embeddings(text):
  tokens = tokenizer(text, return_tensors="pt", padding=True, max_length=128,truncation=True)
  with torch.no_grad():
    outputs = text_model(**tokens)
    embeddings = outputs.last_hidden_state.squeeze(0)
  return embeddings

# print(tokenizer.vocab_size)
# print(len(tokenizer))
