import torch
import torch.nn as nn
import torch.optim as optim

# 自定义神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(2,2)
        self.fc2 = nn.Linear(2,1)

    # 前向传播
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()

# 损失函数
criterion = nn.MSELoss()
# 优化器
optimizer = optim.Adam(model.parameters(),lr=0.001)

X = torch.randn(10,2)
Y = torch.randn(10,1)

# 轮数
for epoch in range(100):
    # 训练模式
    model.train()
    # 清除梯度
    optimizer.zero_grad()
    # 前向传播
    output = model(X)
    # 计算损失
    loss = criterion(output,Y)
    # 反向传播
    loss.backward()
    # 更新权重
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"轮数[{epoch + 1} / 100],损失率:{loss.item():.4f}")
