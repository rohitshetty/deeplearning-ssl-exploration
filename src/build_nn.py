import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Source: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork (nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        # ReLU is an activation function - rectified linear unit. 
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), 
            nn.ReLU(), 
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)

print("logits",logits)
# This should create a list with probability, totalling to 1
prediction_probability = nn.Softmax(dim=1)(logits)

print("prob_pred:", prediction_probability)

# We then pick out one value that is maximised 
y_pred= prediction_probability.argmax(1)

print(f"Predicted class: {y_pred}")