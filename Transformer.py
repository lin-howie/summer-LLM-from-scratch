import torch
import torch.nn as nn
from Classes import MultiHeadAttention, GELU, FeedForward, LayerNorm

GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary Size
    "context_length": 1024, # Context Length
    "emb_dim": 768, # Embedding Dimension
    "n_heads": 12, # Number of Attention Heads
    "n_layers": 12, # Number of layers (transformer blocks)
    "drop_rate": 0.1, # Dropout Rate
    "qkv_bias": False, # Query-Key-Value Bias
}
                                   
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        # Shortcut connection for attention block
        # The Pre-normalization layer
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x) # Shape [batch_size, num_tokens, emb_size]
        self.drop_shortcut(x)
        x = x + shortcut # Add the original input back

        # shortcut connections for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        # 2x4x768
        self.drop_shortcut(x)
        x = x + shortcut # add the original input back

        return x
        # 2x4x768

"""
torch.manual_seed(123)
x = torch.rand(2, 4, 768) #A
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
"""
