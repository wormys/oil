import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch
from utils.handler import getData_task
from model.net import Net
import pandas as pd
import matplotlib.pyplot as plt
import csv
plt.rcParams['font.sans-serif']='Times New Roman'
plt.rcParams['font.size']=18

def main():
    train_file = 'data/fracture_20200617.csv'
    test_file = 'data/fracture_20200627.csv'
    save_train_file = 'data/output/task1_train_loss.csv'
    save_test_file = 'data/output/task1_test_loss.csv'
    test_loss = []
    train_loss = []
    train_Loader,test_Loader = getData_task(train_file, test_file)
    #print(test_Loader)
    net = Net(4, 100, 50, 17)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()
    net.train()
    for epoch in range(10):
        for step,data in enumerate(train_Loader):
            x=data[:, :4]
            y=data[:, 5:]
            prediction = net(x)
            loss = loss_func(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # print(train_loss)
    # save only the parameters
    torch.save(net.state_dict(), 'pkl/task1_params.pkl')
    net.eval()
    MSE = 0
    for step, data1 in enumerate(test_Loader):
        x = data1[:, :4]
        y = data1[:, 5:]
        prediction = net(x)
        # print(prediction,prediction.shape)
        # # Visualization
        # pca=PCA(n_components=1)
        # newX=pca.fit_transform(x).reshape(100)
        # #print(y,prediction)
        # print(pca.explained_variance_ratio_)
        MSE = loss_func(prediction, y)
        test_loss.append(MSE.item())
    to_csv(save_train_file, train_loss)
    to_csv(save_test_file, test_loss)

def to_csv(filename,data):
    with open(filename,'w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['step','loss'])
        for i,rows in enumerate(data):
            writer.writerow([i+1,rows])

def figure_plot():
    train_data = pd.read_csv('data/output/task1_train_loss.csv')
    test_data = pd.read_csv('data/output/task1_test_loss.csv')

    #train figure
    plt.plot(np.array(train_data)[:, 0], np.array(train_data)[:, 1], 'r-', lw=2)
    plt.xlabel("Global_Step")
    plt.ylabel('Loss')
    plt.xlim(-0.5, 1000)
    plt.savefig('figure/task1_loss.png', bbox_inches='tight')
    plt.show()


    # test
    plt.plot(np.array(test_data)[:, 0], np.array(test_data)[:, 1], 'r-', lw=2)
    plt.xlabel("Batch_Numbers")
    plt.ylabel('MSE')
    plt.xlim(-0.5, 35)
    plt.savefig('figure/task1_test_MSE.png',bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    main()
    figure_plot()
