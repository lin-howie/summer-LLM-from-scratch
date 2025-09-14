# Needs to be in MiniLLM folder to work
# Originally from L11PositionalEmbedding
import torch
from L8L9GPT.InputTargetPairs import create_dataloader_v1


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
output_dim = 256
max_length = 4

# DataLoader
dataloader = create_dataloader_v1(raw_text, 
batch_size=8, 
max_length=max_length, 
stride=max_length, 
shuffle=False, 
drop_last=True, 
num_workers=0)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs: \n", inputs)
print("\nInputs shape:\n", inputs.shape)

# Creating the token embeddings
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)

print(token_embeddings.shape)

# Absolute Positional Embedding
context_len = max_length
pos_embedding_layer = torch.nn.Embedding(context_len, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))

print(pos_embeddings.shape)

# Adding the values to create the input embeddings
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)