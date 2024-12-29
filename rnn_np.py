import numpy as np


# 激活函数，这里使用tanh
def tanh(x):
    return np.tanh(x)


# RNN单元类
class RNNCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 初始化权重
        self.W_ih = np.random.randn(hidden_size, input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size)
        self.b_h = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev):
        x = x.reshape(-1, 1)  # 确保x是列向量
        # 计算隐藏状态
        h_t = tanh(np.dot(self.W_ih, x) + np.dot(self.W_hh, h_prev) + self.b_h)
        return h_t

# 简单示例，字符到索引的映射
char_to_idx = {'a': 0, 'b': 1}
idx_to_char = {0: 'a', 1: 'b'}

# 输入序列
input_sequence = np.array([[char_to_idx['a']], [char_to_idx['b']]])

# 初始化RNN
rnn = RNNCell(input_size=2, hidden_size=2)
h_prev = np.zeros((2, 1))

# 逐时间步处理输入序列
hidden_states = []
for x in input_sequence:
    h_prev = rnn.forward(x, h_prev)
    hidden_states.append(h_prev)

print("隐藏状态序列:")
for state in hidden_states:
    print(state.flatten())