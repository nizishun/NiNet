import torch
import torch.nn as nn
from modules.T2Block import ST2B


class expF(nn.Module):
    def __init__(self):
        super().__init__()
        self.clamp = nn.Parameter(torch.randn(1))

    def forward(self, s):
        return torch.exp(self.clamp * (torch.sigmoid(s) - 0.5))


class IMM_block(nn.Module):
    def __init__(self, input_C):
        super().__init__()

        self.f1 = ST2B(5*input_C,input_C,middle_C=4*input_C)
        self.f2 = ST2B(5*input_C,input_C,middle_C=4*input_C)
        self.f3 = ST2B(5*input_C,input_C,middle_C=4*input_C)
        self.f4 = ST2B(5*input_C,input_C,middle_C=4*input_C)
        self.f5 = ST2B(5*input_C,input_C,middle_C=4*input_C)
        self.f6 = ST2B(5*input_C,input_C,middle_C=4*input_C)
        self.fx = ST2B(6*input_C,3*input_C,middle_C=6*input_C)
        self.e1 = expF()
        self.e3 = expF()
        self.e5 = expF()

    def forward(self, x, y1, y2, y3, rev=False, repair=True):

        if not rev:
            y1 = y1 * self.e1(self.f1(torch.cat((x, y2, y3), 1))) + self.f2(torch.cat((x, y2, y3), 1))
            y2 = y2 * self.e3(self.f3(torch.cat((x, y1, y3), 1))) + self.f4(torch.cat((x, y1, y3), 1))
            y3 = y3 * self.e5(self.f5(torch.cat((x, y1, y2), 1))) + self.f6(torch.cat((x, y1, y2), 1))

        elif repair:
            x = self.fx(torch.cat((x, y1, y2, y3), 1))
            y3 = (y3 - self.f6(torch.cat((x, y1, y2), 1))) / self.e5(self.f5(torch.cat((x, y1, y2), 1)))
            y2 = (y2 - self.f4(torch.cat((x, y1, y3), 1))) / self.e3(self.f3(torch.cat((x, y1, y3), 1)))
            y1 = (y1 - self.f2(torch.cat((x, y2, y3), 1))) / self.e1(self.f1(torch.cat((x, y2, y3), 1)))

        else:
            y3 = (y3 - self.f6(torch.cat((x, y1, y2), 1))) / self.e5(self.f5(torch.cat((x, y1, y2), 1)))
            y2 = (y2 - self.f4(torch.cat((x, y1, y3), 1))) / self.e3(self.f3(torch.cat((x, y1, y3), 1)))
            y1 = (y1 - self.f2(torch.cat((x, y2, y3), 1))) / self.e1(self.f1(torch.cat((x, y2, y3), 1)))
        return y1, y2, y3

