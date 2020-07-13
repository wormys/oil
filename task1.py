import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch
from utils.handler import getData_task1
from model.net import Net


def main():
    train_file = 'data/fracture_20200617.csv'
    test_file = 'data/fracture_20200627.csv'
    trainXLoader, trainYLoader, testXLoader, testYloader = getData_task1(train_file, test_file)
    print(testXLoader)
    net = Net(4, 100, 50, 17)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()
    for epoch in range(100):
        for step in range(50):
            for x, y in zip(trainXLoader, trainYLoader):
                prediction = net(x)
                loss = loss_func(prediction, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if epoch % 5 == 0:
            print('loss:', loss)


if __name__ == '__main__':
    main()
