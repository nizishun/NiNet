import torch
import torch.nn as nn
from modules.IMMblock import IMM_block


class IMM_IRM_net(nn.Module):

    def __init__(self, input_C):
        super(IMM_IRM_net, self).__init__()

        self.input_C=int(input_C/3)

        self.inv1 = IMM_block(input_C=self.input_C)
        self.inv2 = IMM_block(input_C=self.input_C)
        self.inv3 = IMM_block(input_C=self.input_C)
        self.inv4 = IMM_block(input_C=self.input_C)
        self.inv5 = IMM_block(input_C=self.input_C)
        self.inv6 = IMM_block(input_C=self.input_C)
        self.inv7 = IMM_block(input_C=self.input_C)
        self.inv8 = IMM_block(input_C=self.input_C)

        self.inv9 = IMM_block(input_C=self.input_C)
        self.inv10 = IMM_block(input_C=self.input_C)
        self.inv11 = IMM_block(input_C=self.input_C)
        self.inv12 = IMM_block(input_C=self.input_C)
        self.inv13 = IMM_block(input_C=self.input_C)
        self.inv14 = IMM_block(input_C=self.input_C)
        self.inv15 = IMM_block(input_C=self.input_C)
        self.inv16 = IMM_block(input_C=self.input_C)

        self.inv17 = IMM_block(input_C=self.input_C)
        self.inv18 = IMM_block(input_C=self.input_C)
        self.inv19 = IMM_block(input_C=self.input_C)
        self.inv20 = IMM_block(input_C=self.input_C)

    def forward(self, x, y, rev=False, repair=True):
        y1, y2, y3 = (y.narrow(1, 0, self.input_C),
                      y.narrow(1, self.input_C, self.input_C),
                      y.narrow(1, 2 * self.input_C, self.input_C))

        if not rev:
            y1, y2, y3 = self.inv1(x, y1, y2, y3)
            y1, y2, y3 = self.inv2(x, y1, y2, y3)
            y1, y2, y3 = self.inv3(x, y1, y2, y3)
            y1, y2, y3 = self.inv4(x, y1, y2, y3)
            y1, y2, y3 = self.inv5(x, y1, y2, y3)
            y1, y2, y3 = self.inv6(x, y1, y2, y3)
            y1, y2, y3 = self.inv7(x, y1, y2, y3)
            y1, y2, y3 = self.inv8(x, y1, y2, y3)

            y1, y2, y3 = self.inv9(x, y1, y2, y3)
            y1, y2, y3 = self.inv10(x, y1, y2, y3)
            y1, y2, y3 = self.inv11(x, y1, y2, y3)
            y1, y2, y3 = self.inv12(x, y1, y2, y3)
            y1, y2, y3 = self.inv13(x, y1, y2, y3)
            y1, y2, y3 = self.inv14(x, y1, y2, y3)
            y1, y2, y3 = self.inv15(x, y1, y2, y3)
            y1, y2, y3 = self.inv16(x, y1, y2, y3)

            y1, y2, y3 = self.inv17(x, y1, y2, y3)
            y1, y2, y3 = self.inv18(x, y1, y2, y3)
            y1, y2, y3 = self.inv19(x, y1, y2, y3)
            y1, y2, y3 = self.inv20(x, y1, y2, y3)

        elif repair:
            y1, y2, y3 = self.inv20(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv19(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv18(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv17(x, y1, y2, y3, rev=True)

            y1, y2, y3 = self.inv16(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv15(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv14(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv13(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv12(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv11(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv10(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv9(x, y1, y2, y3, rev=True)

            y1, y2, y3 = self.inv8(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv7(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv6(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv5(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv4(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv3(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv2(x, y1, y2, y3, rev=True)
            y1, y2, y3 = self.inv1(x, y1, y2, y3, rev=True)

        else:
            y1, y2, y3 = self.inv20(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv19(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv18(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv17(x, y1, y2, y3, rev=True, repair=False)

            y1, y2, y3 = self.inv16(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv15(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv14(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv13(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv12(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv11(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv10(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv9(x, y1, y2, y3, rev=True, repair=False)

            y1, y2, y3 = self.inv8(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv7(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv6(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv5(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv4(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv3(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv2(x, y1, y2, y3, rev=True, repair=False)
            y1, y2, y3 = self.inv1(x, y1, y2, y3, rev=True, repair=False)
        
        return torch.cat((y1, y2, y3), 1)


