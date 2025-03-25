import torch
import math
from otdd.pytorch.utils import generate_moments
import torch.nn as nn

num_channels = 3
num_projection = 100
model = nn.Sequential(
    nn.Conv2d(num_channels, num_projection, kernel_size=3, stride=2, padding=1, bias=False),
    nn.Conv2d(num_projection, num_projection, kernel_size=3, stride=2, padding=1, bias=False, groups=num_projection),
    nn.Conv2d(num_projection, num_projection, kernel_size=3, stride=2, padding=1, bias=False, groups=num_projection),
    nn.Conv2d(num_projection, num_projection, kernel_size=3, stride=2, padding=1, bias=False, groups=num_projection),
    nn.Conv2d(num_projection, num_projection, kernel_size=3, stride=2, padding=1, bias=False, groups=num_projection),
    nn.Conv2d(num_projection, num_projection, kernel_size=7, stride=1, padding=0, bias=False, groups=num_projection)
)

input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(output.shape)