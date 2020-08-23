import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import csv
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
BATCH_SIZE = 50

FONTSIZE=18


def getData_task(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # get all features
    features = np.array(train_data.columns)

    # Z-score normalization
    scaler=MinMaxScaler()
    data1 = np.array(scaler.fit_transform(train_data[features]),dtype=np.float32)
    data2 = np.array(scaler.fit_transform(test_data[features]), dtype=np.float32)
    print(data1)

    train_Loader=DataLoader(data1,batch_size=BATCH_SIZE,shuffle=True)
    test_Loader=DataLoader(data2,batch_size=BATCH_SIZE,shuffle=True)

    return train_Loader, test_Loader



# def main():
#     getData_task1("../data/fracture_20200617.csv","../data/fracture_20200627.csv")
#
# if __name__=='__main__':
#     main()
