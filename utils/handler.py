import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import csv
from torch.autograd import Variable

BATCH_SIZE = 50


def getData_task1(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # get all features
    features = np.array(train_data.columns)

    # Z-score normalization
    train_data[features] = train_data[features].apply(lambda x: (x - x.mean()) / (x.std()))
    test_data[features] = test_data[features].apply(lambda x: (x - x.mean()) / (x.std()))

    # 2numpy
    data1 = np.array(train_data, dtype=np.float32)
    data2 = np.array(test_data, dtype=np.float32)
    print(data1)

    trainX_data = data1[:, :4]
    trainY_data = data1[:, 5:]
    # print(trainX_data,trainY_data.shape)

    testX_data = data2[:, :4]
    testY_data = data2[:, 5:]

    # dataloader
    trainXLoader = DataLoader(trainX_data, batch_size=BATCH_SIZE, shuffle=True)
    trainYLoader = DataLoader(trainY_data, batch_size=BATCH_SIZE, shuffle=True)
    testXLoader = DataLoader(testX_data, batch_size=BATCH_SIZE, shuffle=True)
    testYLoader = DataLoader(testY_data, batch_size=BATCH_SIZE, shuffle=True)

    return trainXLoader, trainYLoader, testXLoader, testYLoader


def getData_task2(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    # get all features
    features = np.array(train_data.columns)

    # Z-score normalization
    train_data[features] = train_data[features].apply(lambda x: (x - x.mean()) / (x.std()))
    test_data[features] = test_data[features].apply(lambda x: (x - x.mean()) / (x.std()))
    # print(train_data)

    # 2numpy
    data1 = np.array(train_data, dtype=np.float32)
    data2 = np.array(test_data, dtype=np.float32)
    # print(data)

    trainX_data = data1[:, 5:]
    trainY_data = data1[:, 4]
    # print(trainX_data,trainY_data)

    testX_data = data2[:, 5:]
    testY_data = data2[:, 4]

    # dataloader
    trainXLoader = DataLoader(trainX_data, batch_size=BATCH_SIZE, shuffle=True)
    trainYLoader = DataLoader(trainY_data, batch_size=BATCH_SIZE, shuffle=True)

    testXLoader = DataLoader(testX_data, batch_size=BATCH_SIZE, shuffle=True)
    testYLoader = DataLoader(testY_data, batch_size=BATCH_SIZE, shuffle=True)

    return trainXLoader, trainYLoader, testXLoader, testYLoader

# def main():
#     getData_task1()
#
# if __name__=='__main__':
#     main()
