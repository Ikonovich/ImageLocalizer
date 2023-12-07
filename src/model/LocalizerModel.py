import torch
from torch import nn
import torch.nn.functional as F


class Localizer(nn.Module):

    def __init__(self):
        super(Localizer, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, device="cuda", dtype=torch.float32)
        self.pool = nn.MaxPool2d(3, 3)
        self.flatten = nn.Flatten(0, -1)
     #   self.conv2 = nn.Conv2d(6, 16, 5, device="cuda", dtype=torch.float32)
        self.fc1 = nn.Linear(4435872, 120, device="cuda", dtype=torch.float32)
        self.fc2 = nn.Linear(120, 84, device="cuda", dtype=torch.float32)
        self.fc3 = nn.Linear(84, 4, device="cuda", dtype=torch.float32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
     #   x = self.pool(x)
       # x = F.relu(self.conv2(x))
      #  x = self.pool(x)
        # x = x.view(-1, 16 * 4 * 4)
        x = self.flatten(x)
        x = F.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    #     self.convIn = nn.Conv2d(3, 32, 3, 1, device="cuda", dtype=torch.float32)
    #
    #     self.flatten = nn.Flatten(0, -1)
    #     self.linear1 = nn.LazyLinear(1000, True, device="cuda", dtype=torch.float32)
    #     self.linearOut = nn.LazyLinear(4, True, device="cuda", dtype=torch.float32)
    #
    # def forward(self, x):
    #     x = self.convIn(x)
    #     x = self.flatten(x)
    #     x = F.relu(x)
    #     x = self.linear1(x)
    #     x = F.relu(x)
    #     output = F.relu(self.linearOut(x))
    #     return output

