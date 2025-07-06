import torch
import torch.nn as nn
from modules.C3ITblock import C3IT_block


class C3IT_IC3IT_net(nn.Module):

    def __init__(self, input_C):
        super(C3IT_IC3IT_net, self).__init__()

        self.input_C=int(input_C/3)

        self.inv1 = C3IT_block(input_C=self.input_C)
        self.inv2 = C3IT_block(input_C=self.input_C)
        # self.inv3 = C3IT_block(input_C=self.input_C)
        # self.inv4 = C3IT_block(input_C=self.input_C)
        # self.inv5 = C3IT_block(input_C=self.input_C)
        # self.inv6 = C3IT_block(input_C=self.input_C)
        # self.inv7 = C3IT_block(input_C=self.input_C)
        # self.inv8 = C3IT_block(input_C=self.input_C)

        # self.inv9 = C3IT_block(input_C=self.input_C)
        # self.inv10 = C3IT_block(input_C=self.input_C)
        # self.inv11 = C3IT_block(input_C=self.input_C)
        # self.inv12 = C3IT_block(input_C=self.input_C)
        # self.inv13 = C3IT_block(input_C=self.input_C)
        # self.inv14 = C3IT_block(input_C=self.input_C)
        # self.inv15 = C3IT_block(input_C=self.input_C)
        # self.inv16 = C3IT_block(input_C=self.input_C)


    def forward(self, x, rev=False):
        x1, x2, x3 = (x.narrow(1, 0, self.input_C),
                      x.narrow(1, self.input_C, self.input_C),
                      x.narrow(1, 2 * self.input_C, self.input_C))

        if not rev:
            x1, x2, x3 = self.inv1(x1, x2, x3)
            x1, x2, x3 = self.inv2(x1, x2, x3)
            # x1, x2, x3 = self.inv3(x1, x2, x3)
            # x1, x2, x3 = self.inv4(x1, x2, x3)
            # x1, x2, x3 = self.inv5(x1, x2, x3)
            # x1, x2, x3 = self.inv6(x1, x2, x3)
            # x1, x2, x3 = self.inv7(x1, x2, x3)
            # x1, x2, x3 = self.inv8(x1, x2, x3)

            # x1, x2, x3 = self.inv9(x1, x2, x3)
            # x1, x2, x3 = self.inv10(x1, x2, x3)
            # x1, x2, x3 = self.inv11(x1, x2, x3)
            # x1, x2, x3 = self.inv12(x1, x2, x3)
            # x1, x2, x3 = self.inv13(x1, x2, x3)
            # x1, x2, x3 = self.inv14(x1, x2, x3)
            # x1, x2, x3 = self.inv15(x1, x2, x3)
            # x1, x2, x3 = self.inv16(x1, x2, x3)

        else:
            # x1, x2, x3 = self.inv16(x1, x2, x3, rev=True)
            # x1, x2, x3 = self.inv15(x1, x2, x3, rev=True)
            # x1, x2, x3 = self.inv14(x1, x2, x3, rev=True)
            # x1, x2, x3 = self.inv13(x1, x2, x3, rev=True)
            # x1, x2, x3 = self.inv12(x1, x2, x3, rev=True)
            # x1, x2, x3 = self.inv11(x1, x2, x3, rev=True)
            # x1, x2, x3 = self.inv10(x1, x2, x3, rev=True)
            # x1, x2, x3 = self.inv9(x1, x2, x3, rev=True)

            # x1, x2, x3 = self.inv8(x1, x2, x3, rev=True)
            # x1, x2, x3 = self.inv7(x1, x2, x3, rev=True)
            # x1, x2, x3 = self.inv6(x1, x2, x3, rev=True)
            # x1, x2, x3 = self.inv5(x1, x2, x3, rev=True)
            # x1, x2, x3 = self.inv4(x1, x2, x3, rev=True)
            # x1, x2, x3 = self.inv3(x1, x2, x3, rev=True)
            x1, x2, x3 = self.inv2(x1, x2, x3, rev=True)
            x1, x2, x3 = self.inv1(x1, x2, x3, rev=True)
        
        return torch.cat((x1, x2, x3), 1)


