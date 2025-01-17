import torch
from torch import nn
import os

class QNetwork(nn.Module):
    def __init__(self, num_hidden=128):
        super().__init__()
        self.l1 = nn.Linear(5, num_hidden)
        self.l2 = nn.Linear(num_hidden, 5)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))

model_path = "model.pth"

# Load state dict and compare
model = QNetwork()
raw_state_dict = torch.load(model_path, map_location=torch.device("cpu"))

# Verify keys
print("Keys in model state dict:", model.state_dict().keys())
print("Keys in loaded state dict:", raw_state_dict.keys())

# Attempt strict loading
try:
    model.load_state_dict(raw_state_dict, strict=True)
    print("Loaded state dict with strict=True.")
except RuntimeError as e:
    print(f"Error in loading state dict: {e}")

# Compare values
for name, param in model.state_dict().items():
    if name in raw_state_dict:
        diff = torch.sum(torch.abs(param - raw_state_dict[name]))
        print(f"Difference for '{name}': {diff.item()}")

# Check for zero parameters
all_zero = True
for name, param in model.state_dict().items():
    print(f"Parameter '{name}' values:\n{param}")
    is_zero = torch.allclose(param, torch.zeros_like(param), atol=1e-8)
    print(f"Parameter '{name}' is all zero: {is_zero}")
    if not is_zero:
        all_zero = False

if all_zero:
    print("All weights and biases are zero.")
else:
    print("Not all weights and biases are zero.")
