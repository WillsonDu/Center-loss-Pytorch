import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),

            nn.Linear(64, 10),
            nn.BatchNorm1d(10),
            nn.PReLU()
        )

        self.fc1 = nn.Linear(10, 2)
        self.fc2 = nn.Linear(2, 10)

    def forward(self, x):
        x = self.conv1(x)  # batch,10
        features = self.fc1(x)  # batch,2  #记得把导数第二层的输出(可加激活函数, 形状为[batch,2])返回
        labels = self.fc2(features)  # batch,10
        return features, labels


if __name__ == '__main__':
    net = Net()
    x = torch.Tensor(np.arange(1, 100 * 784 + 1).reshape((100, 784)))
    x, y = net(x)
    print(x)
    print(y)
