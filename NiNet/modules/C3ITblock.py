import torch
import torch.nn as nn
from modules.T2Block import ST2B



class expF(nn.Module):
    def __init__(self):
        super().__init__()
        self.clamp = nn.Parameter(torch.randn(1))

    def forward(self, s):
        return torch.exp(self.clamp * (torch.sigmoid(s) - 0.5))
    

class C3IT_block(nn.Module):
    def __init__(self, input_C):
        super().__init__()

        self.g1 = ST2B(2*input_C,input_C,middle_C=2*input_C)
        self.g2 = ST2B(2*input_C,input_C,middle_C=2*input_C)
        self.g3 = ST2B(2*input_C,input_C,middle_C=2*input_C)
        self.g4 = ST2B(2*input_C,input_C,middle_C=2*input_C)
        self.g5 = ST2B(2*input_C,input_C,middle_C=2*input_C)
        self.g6 = ST2B(2*input_C,input_C,middle_C=2*input_C)
        self.e1 = expF()
        self.e3 = expF()
        self.e5 = expF()

    def forward(self, x1, x2, x3, rev=False):
        if not rev:
            x1 = x1 * self.e1(self.g1(torch.cat((x2, x3), dim=1))) + self.g2(torch.cat((x2, x3), dim=1))
            x2 = x2 * self.e3(self.g3(torch.cat((x1, x3), dim=1))) + self.g4(torch.cat((x1, x3), dim=1))
            x3 = x3 * self.e5(self.g5(torch.cat((x1, x2), dim=1))) + self.g6(torch.cat((x1, x2), dim=1))

        else:
            x3 = (x3 - self.g6(torch.cat((x1, x2), dim=1))) / self.e5(self.g5(torch.cat((x1, x2), dim=1)))
            x2 = (x2 - self.g4(torch.cat((x1, x3), dim=1))) / self.e3(self.g3(torch.cat((x1, x3), dim=1)))
            x1 = (x1 - self.g2(torch.cat((x2, x3), dim=1))) / self.e1(self.g1(torch.cat((x2, x3), dim=1)))

        return x1, x2, x3

