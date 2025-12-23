import torch
import torch.nn as nn


class SimpleNet(nn.modules):
    def __init__(self):
        super.__init__(SimpleNet, self)
        self.layer1 = nn.Linear(1,20)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.layer2 = nn.Linear(20,1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.leaky_relu(x)
        x = self.layer2(x)





