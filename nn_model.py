# CNN model
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable # for computational graphs

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
		# class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
		# padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(1, 6, 5, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
