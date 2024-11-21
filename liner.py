
# import matplotlib 
# matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd

sns.set_style(style = 'whitegrid')
plt.rcParams["patch.force_edgecolor"] = True



m = 2 # slope
c = 3 # interceptm = 2 # slope
# c = 3 # intercept
x = np.random.rand(256)
noise = np.random.randn(256) / 4
y = x * m + c + noise
df = pd.DataFrame()
df['x'] = x
df['y'] = y
sns.lmplot(x ='x', y ='y', data = df)
plt.savefig("./Linering.png")


##    上面是线性回归拟合的，下面用一个线形层 拟合的结果
import torch
import torch.nn as nn
from torch.autograd import Variable
x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')

class LinearRegressionModel(nn.Module):
   def __init__(self, input_dim, output_dim):
      super(LinearRegressionModel, self).__init__()
      self.linear = nn.Linear(input_dim, output_dim)
   def forward(self, x):
      out = self.linear(x)
      return out

input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
#input_dim, output_dim(1, 1)
model = LinearRegressionModel(input_dim, output_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


def get_param_values():
       return model.parameters()

def plot_current_fit(title = ""):
   plt.figure(figsize = (12,4))
   plt.title(title)
   plt.scatter(x, y, s = 8) # 绘制散点图
   w, b = get_param_values()
   w1 = w.data[0][0].numpy()
   b1 = b.data[0].numpy()
   x1 = np.array([0., 1.])
   y1 = x1 * w1 + b1
   plt.plot(x1, y1, 'r', label = 'Current Fit ({:.3f}, {:.3f})'.format(w1, b1))
   plt.xlabel('x (input)')
   plt.ylabel('y (target)')
   plt.legend()
 #plt.show()
   plt.savefig("./"+title+".png")

plot_current_fit('Before training')

#训练迭代
for echo in range(500):
       y_pred = model(Variable(torch.from_numpy(x_train)))
       loss = criterion(y_pred, Variable(torch.from_numpy(y_train)))
       print("Epoch {}, Loss {}".format(echo, loss.item()))
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

plot_current_fit('After training')



