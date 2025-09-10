GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary Size
    "context_length": 256, # Context Length
    "emb_dim": 768, # Embedding Dimension
    "n_heads": 12, # Number of Attention Heads
    "n_layers": 12, # Number of layers (transformer blocks)
    "drop_rate": 0.1, # Dropout Rate
    "qkv_bias": False, # Query-Key-Value Bias
}

from GPTModel import GPTModel, generate_text_simple
from Classes import MultiHeadAttention
from Transformer import TransformerBlock
import torch
import tiktoken

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG_124M)

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

# Generate tokens using your model
out = generate_text_simple(model, encoded_tensor, max_new_tokens=20, context_size=128)

# Flatten and decode
tokens = out.view(-1).tolist()
decoded_text = tokenizer.decode(tokens)

print("decoded_text:", decoded_text)
