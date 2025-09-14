import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts (x^3)
     [0.22, 0.58, 0.33], # with (x^4)
     [0.77, 0.25, 0.10], # one (x^5)
     [0.05, 0.80, 0.55]] # step (x^6)]
)

# Getting query, key, and value weights
x_2 = inputs[1] #A
d_in = inputs.shape[1] #B Also d_in is basiclaly equal to 3 because the dimensions are 3 and we are doing matrix multiplication
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print(W_query)
print(W_key)
print(W_value)

# Getting query, key, and value matrix
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2)

# Getting queries, keys, and values vectors
queries = inputs @ W_query
keys = inputs @ W_key
values = inputs @ W_value

print("Queries.shape: ", queries.shape)
print("Keys.shape: ", keys.shape)
print("Values.shape: ", values.shape)

# To find the attention score for the second query, 
# we need to take the dot product of the second query vector and all the key vectors
keys_2 = keys[1] #A
query_2 = queries[1]

attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

attn_score_2 = query_2 @ keys.T # All attention scores for query 2
print (attn_score_2)

attn_scores = queries @ keys.T # Omega

print(attn_scores)

# Creating the context vector for second word/token
# READ THE NOTE AT THE BOTTOM
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_score_2 / d_k**.5, dim=-1)
print(attn_weights_2)
print(d_k)
# NOTE we scale by d_keys to maintian STABILITY IN LEARNING
# Without it, there might be uneven peaks in data if the data is large
# The high values become very pronounces
# This makes the softmax output to become "peaky" where the highest values receive all the probability mass
# Sharp learning makes learning unstable

# NOTE There's more! We uses square root because it is related to the variance produced from two random numbers (Q and K)
# Variance increases with dimensions
# Dividing by the square root of dimensions, sqrt(dimensions), keeps the variance close to 1

# Creating the attention weights for every word/token
# NOTE this is called SCALED DOT PRODUCT ATTENTION
attn_weights = torch.softmax(attn_scores / d_k**.5, dim=-1)
print(attn_weights)

# Creating context vector(s)
context_vec_2 = attn_weights_2 @ values

print(context_vec_2)


# Trying out the new class
from SelfAttention import SelfAttention_v1

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
