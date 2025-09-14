import torch
from SelfAttention import SelfAttention_v2
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts (x^3)
     [0.22, 0.58, 0.33], # with (x^4)
     [0.77, 0.25, 0.10], # one (x^5)
     [0.05, 0.80, 0.55]] # step (x^6)]
)
x_2 = inputs[1] #A
d_in = inputs.shape[1] #B Also d_in is basiclaly equal to 3 because the dimensions are 3 and we are doing matrix multiplication
d_out = 2

sa_v2 = SelfAttention_v2(d_in, d_out)

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)

attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

# Demonstrating a simple version of masking
# triu is upper triangular matrix (lower triangle is 0)
# tril is lower triangular matrix (upper triangle is 0)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

masked_simple = attn_weights*mask_simple
print(masked_simple)

row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
# NOTE The issue with this method is that we took softmax of the attn_socres before so the re-normalized values 
# are still affected by the other values. This leads to data leakage (future data is leaking into the beginning data)

# SOLUTION: applying an upper triangular infinity mask
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

# Example for Dropout
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))

print(dropout(attn_weights))

# Now we are creating code to account for multiple batches of data
batch = torch.stack((inputs, inputs), dim=0) # 2 inputs, 6 tokens, 3 dimensions per token
print(batch.shape)


from CausalAttention import CausalAttention

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vec = ca(batch)
print("context_vec.shape: ", context_vec.shape)


