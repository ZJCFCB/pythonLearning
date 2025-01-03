from torch.autograd import Variable
import torch.nn.functional as F


class SimpleCNN(torch.nn.Module):
    def __init__(self):
      super(SimpleCNN, self).__init__()
      #Input channels = 3, output channels = 18
      self.conv1 = torch.nn.Conv2d(3, 18, kernel_size = 3, stride = 1, padding = 1)
      self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
      #4608 input features, 64 output features (see sizing flow below)
      self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
      #64 input features, 10 output features for our 10 defined classes
      self.fc2 = torch.nn.Linear(64, 10)
    def forward(self, x):
       x = F.relu(self.conv1(x))
       x = self.pool(x)
       x = x.view(-1, 18 * 16 *16)
       x = F.relu(self.fc1(x))
      #Computes the second fully connected layer (activation applied later)
      #Size changes from (1, 64) to (1, 10)
       x = self.fc2(x)
       return(x)