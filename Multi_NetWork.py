import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        self.drop = nn.Dropout(0.5)
        self.fc21 = nn.Linear(150, 2)
        self.fc22 = nn.Linear(150, 3)

        self.soft1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = F.relu(x)
        x = self.pool2(self.conv2(x))
#         x = self.drop(x)
        x = F.relu(x)

        x = x.view(-1, 6 * 123 * 123)
        x = self.fc1(x)
        x = self.drop(x)
        x = F.relu(x)
        x1 = self.fc21(x)
        x2 = self.fc22(x)
        x1 = self.soft1(x1)
        x2 = self.soft1(x2)

        return x1, x2

    
class Net_BCE(nn.Module):
    def __init__(self):
        super(Net_BCE, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(150, 5)

        self.sig=nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = F.relu(x)
        x = self.pool2(self.conv2(x))
        x = F.relu(x)

        x = x.view(-1, 6 * 123 * 123)
        x = self.fc1(x)
        x = self.drop(x)
        x = F.relu(x)
        x = self.fc2(x)
        x=self.sig(x)

        return x
