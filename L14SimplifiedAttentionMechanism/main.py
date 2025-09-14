import torch

# Creating first attention score using the second token in the given input
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts (x^3)
     [0.22, 0.58, 0.33], # with (x^4)
     [0.77, 0.25, 0.10], # one (x^5)
     [0.05, 0.80, 0.55]] # step (x^6)]
)

query = inputs[1] # The second token

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)

# Normalization (creating attention weights. Bascially making the values sum up to 1)
attn_weights_2_tmp = attn_scores_2/attn_scores_2.sum()

print("Attention Weights: ", attn_weights_2_tmp)
print("Sum of Attention Weights:", attn_weights_2_tmp.sum())


# Creating a softmax normalization function. 
# Values are always positive because we use exponents (the values have to be positive for easy interpretation)
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention Weights: ", attn_weights_2_naive)
print("Sum: ", attn_weights_2_naive.sum())

# Pytorch version of softmax
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention Weights: ", attn_weights_2)
print("Sum: ", attn_weights_2.sum())

# Creating the Context Vector
query = inputs[1]

context_vec_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)

# Creating an entire context vector input

"""
# This approach is a lot slower because of the double for loops
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)
"""
# A faster method is multiplying the input matrix with the TRANSPOSE of itself
# This is an linear algebra concept
attn_scores = torch.empty(6, 6)
attn_scores = inputs @ inputs.T

print(attn_scores)

# Now softmax normalizing it
# Notice dim=-1 and not dim=0 (I think this is to regress a dimension to columns because this is a two dimensional vector)
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
