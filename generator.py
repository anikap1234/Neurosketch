def generate_code(architecture):
    layers_code = []

    for layer in architecture["layers"]:
        if layer == "conv":
            layers_code.append("nn.Conv2d(3, 64, 3, padding=1)")
        elif layer == "relu":
            layers_code.append("nn.ReLU()")
        elif layer == "pool":
            layers_code.append("nn.MaxPool2d(2)")
        elif layer == "upconv":
            layers_code.append("nn.ConvTranspose2d(64, 32, 2, stride=2)")
        elif layer == "flatten":
            layers_code.append("nn.Flatten()")
        elif layer == "dense":
            layers_code.append("nn.Linear(128, 10)")

    layers_str = ",\n            ".join(layers_code)

    code = f"""
import torch
import torch.nn as nn

class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            {layers_str}
        )

    def forward(self, x):
        return self.model(x)
"""

    return code