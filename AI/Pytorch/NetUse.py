import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 设置数据
n_in,n_h,n_out,batch_size = 10,5,1,10

x = torch.randn(batch_size,n_in)
y = torch.tensor([[1.0], [0.0], [0.0],
                  [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

# 设置简单模型
model = nn.Sequential(
    nn.Linear(n_in,n_h),
    nn.ReLU(),
    nn.Linear(n_h,n_out),
    nn.Sigmoid()
)

# 损失率
criterion = torch.nn.MSELoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

losses = []

for epoch in range(50):
    # 前向传播
    y_pred = model(x)
    loss = criterion(y_pred,y)
    losses.append(loss.item())

    print(f"轮数[{epoch + 1} / 50],损失率:{loss.item():.4f}")

    # 清空优化器内容
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    optimizer.step()

plt.figure(figsize=(8, 5))
plt.plot(range(1, 51), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

# 可视化预测结果与实际目标值对比
y_pred_final = model(x).detach().numpy()  # 最终预测值
y_actual = y.numpy()  # 实际值

plt.figure(figsize=(8, 5))
plt.plot(range(1, batch_size + 1), y_actual, 'o-', label='Actual', color='blue')
plt.plot(range(1, batch_size + 1), y_pred_final, 'x--', label='Predicted', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid()
plt.show()