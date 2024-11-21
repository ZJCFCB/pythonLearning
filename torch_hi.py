import torch
import torch.nn as nn

n_in, n_h, n_out, batch_size = 10, 5, 1, 10

x = torch.randn(batch_size, n_in)
y = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

#构建一个模型
model = nn.Sequential(nn.Linear(n_in, n_h),
   nn.ReLU(),
   nn.Linear(n_h, n_out),
   nn.Sigmoid())

criterion = torch.nn.MSELoss() #损失函数

#SGD梯度下降的优化器
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#训练模型
for epoch in range(5000):
   y_pred = model(x)
   loss = criterion(y_pred, y)
  # print('epoch: ', epoch,' loss: ', loss.item())

   optimizer.zero_grad() #清空梯度信息

   loss.backward()
   optimizer.step()


print(x)
print(model(x))