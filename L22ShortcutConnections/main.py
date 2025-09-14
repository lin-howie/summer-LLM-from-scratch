import torch
import torch.nn as nn
from GELU import GELU

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_size, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_size[0], layer_size[1], GELU())),
            nn.Sequential(nn.Linear(layer_size[1], layer_size[2], GELU())),
            nn.Sequential(nn.Linear(layer_size[2], layer_size[3], GELU())),
            nn.Sequential(nn.Linear(layer_size[3], layer_size[4], GELU())),
            nn.Sequential(nn.Linear(layer_size[4], layer_size[5], GELU()))
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer 
            layer_output = layer(x)
            
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
            
            return x

# Example
layer_size = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_size, use_shortcut=False)

def print_gradient(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Caluclate loss based on how close the target and the output are
    loss = nn.MSELoss()
    loss = loss(output, target)

    # Backward pass
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
            
print_gradient(model_without_shortcut, sample_input)


