import torch
import pandas as pd
import numpy as np


class Net(torch.nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(Net, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input, hidden1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden2, output)

        )

    def forward(self, x):
        x = self.net(x)
        return x
