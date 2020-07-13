import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch
from sklearn.decomposition import PCA
from utils.handler import getData_task2
from model.net import Net
import matplotlib.pyplot as plt
import csv
import os


def main():
    train_file = 'data/fracture_20200617.csv'
    test_file = 'data/fracture_20200627.csv'
    save_file = 'data/output/task2_loss.csv'
    if not os.path.exists(save_file):  # if path not exist, create
        os.makedirs(save_file)
    trainXLoader, trainYLoader, testXLoader, testYloader = getData_task2(train_file, test_file)
    test_loss = []
    train_loss = []
    net = Net(17, 100, 40, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()
    plt.show()
    for epoch in range(100):
        net.train()
        for step in range(100):
            for x, y in zip(trainXLoader, trainYLoader):
                prediction = net(x).reshape(50)
                # print(y,y.shape,prediction.shape)
                # print(y,prediction)
                loss = loss_func(prediction, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_loss.append(loss)
        # print(train_loss)
        net.eval()
        pre_loss = 0
        for x, y in zip(testXLoader, testYloader):
            prediction = net(x).reshape(50)
            # print(prediction,prediction.shape)
            # # Visualization
            # pca=PCA(n_components=1)
            # newX=pca.fit_transform(x).reshape(100)
            # #print(y,prediction)
            # print(pca.explained_variance_ratio_)
            pre_loss = loss_func(prediction, y)
        test_loss.append(pre_loss)
        if epoch % 5 == 0:
            print('train_loss:', loss, ';test_loss:', pre_loss)
    print(train_loss, test_loss)
    to_csv(save_file, train_loss, test_loss)


def to_csv(filename, Matrix1, Matrix2):
    with open(filename, 'a', newline='')as f:
        writer = csv.writer(f)
        headers = ['epoch', 'train_loss', 'test_loss']
        # writer.writerow(headers)
        for i in range(100):
            writer.writerow([i + 1, Matrix1[i], Matrix2[i]])


if __name__ == '__main__':
    main()
