import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 生成一些虚拟的时间序列数据
time_steps = 10
input_size = 1
output_size = 1
batch_size = 1
num_epochs = 100
lr = 0.01

# 生成数据，简单的正弦波
t = np.linspace(0, 20, 200, dtype=np.float32)
data = np.sin(t)
x = []
y = []
for i in range(len(data) - time_steps):
    x.append(data[i:i+time_steps].reshape(-1, input_size))
    y.append(data[i+time_steps].reshape(-1, output_size))

x = np.array(x)
y = np.array(y)

# 转换为张量
x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)


# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


# 初始化模型、损失函数和优化器
hidden_size = 32
model = RNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 训练模型
for epoch in range(num_epochs):
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 预测
with torch.no_grad():
    test_x = x_tensor[-1].unsqueeze(0)
    predict = model(test_x)
    print(f"预测值: {predict.item():.4f}")

# 绘图（可选）
plt.plot(t, data, label='原始数据')
plt.plot(t[-1], predict.item(), 'ro', label='预测点')
plt.legend()
plt.show()