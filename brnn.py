import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator
import numpy as np

# 定义字段
TEXT = Field(tokenize='spacy', lower=True)
LABEL = LabelField(dtype=torch.float)

# 加载数据集
train_data, test_data = IMDB.splits(TEXT, LABEL)

# 构建词汇表
TEXT.build_vocab(train_data, max_size=10000)
LABEL.build_vocab(train_data)

# 创建迭代器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)


# 定义双向RNN模型
class BRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        prediction = self.fc(hidden)
        prediction = self.sigmoid(prediction)
        return prediction


# 初始化模型
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
OUTPUT_DIM = 1
model = BRNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
N_EPOCHS = 3
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_acc = 0

    model.train()
    for batch in train_iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += ((predictions > 0.5) == batch.label.bool()).float().mean().item()

    print(f'Epoch: {epoch + 1}, Train Loss: {train_loss / len(train_iterator)}, Train Acc: {train_acc / len(train_iterator)}')

# 测试模型
test_loss = 0
test_acc = 0

model.eval()
with torch.no_grad():
    for batch in test_iterator:
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)

        test_loss += loss.item()
        test_acc += ((predictions > 0.5) == batch.label.bool()).float().mean().item()

print(f'Test Loss: {test_loss / len(test_iterator)}, Test Acc: {test_acc / len(test_iterator)}')