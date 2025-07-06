import torch
import torch.nn as nn
from modules.SwimTransformerBlock import BasicLayer as STB
from modules.SwimTransformerBlock import PatchDivide, PatchReverse


class ST2B(nn.Module):

    def __init__(self, input_C, output_C, middle_C=4, depth=2, num_heads=8, window_size=8, patch=8):
        super().__init__()
        
        self.patch = patch
        self.inPut = nn.Conv2d(input_C, middle_C, 1, 1)
        self.SwT2B = STB(dim=patch*patch*middle_C, depth=depth, num_heads=num_heads, window_size=window_size)
        self.outPut = nn.Conv2d(middle_C, output_C, 1, 1)


    def forward(self, x):
        _, _, H, W = x.shape
        x = self.inPut(x)
        x = PatchDivide(x, self.patch)
        x = self.SwT2B(x, H//self.patch, W//self.patch)
        x = PatchReverse(x, self.patch)
        x = self.outPut(x)
        return x