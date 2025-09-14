import torch 
import torch.nn as nn
from GELU import GELU

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary Size
    "context_length": 1024, # Context Length
    "emb_dim": 768, # Embedding Dimension
    "n_heads": 12, # Number of Attention Heads
    "n_layers": 12, # Number of layers (transformer blocks)
    "drop_rate": 0.1, # Dropout Rate
    "qkv_bias": False, # Query-Key-Value Bias
}

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), ## Expansion
            GELU(), ## GELU Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), ## Contraction
        )

    def forward(self, x):
        return self.layers(x)

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768) #A
out = ffn(x)
print(out.shape)