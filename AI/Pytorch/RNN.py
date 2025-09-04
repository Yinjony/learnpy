import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

char_set = list("hello")
char_to_idx = {c: i for i, c in enumerate(char_set)}
idx_to_char = {i: c for i, c in enumerate(char_set)}

# 准备数据
input_str = "hello"
target_str = "elloh"
input_data = [char_to_idx[c] for c in input_str]
target_data = [char_to_idx[c] for c in target_str]

# 转换为独热编码
input_one_hot = np.eye(len(char_set))[input_data]

# 得到张量数据
inputs = torch.tensor(input_one_hot,dtype=torch.float32)
targets = torch.tensor(target_data,dtype=torch.long)

# 超参数
input_size = len(char_set)
hidden_size = 8
output_size = len(char_set)
num_epochs = 200
lr = 0.1

# 定义模型
class RNNModel(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNNModel,self).__init__()
        self.rnn = nn.RNN(input_size,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x,hidden):
        # rnn的特有形式
        out, hidden = self.rnn(x,hidden)
        out = self.fc(out)
        return out,hidden

model = RNNModel(input_size,hidden_size,output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

losses = []
# 初始隐藏状态是None
hidden = None
for epoch in range(num_epochs):
    optimizer.zero_grad()

    outputs,hidden = model(inputs.unsqueeze(0),hidden)
    hidden = hidden.detach()#防止梯度爆炸

    loss = criterion(outputs.view(-1,output_size),targets)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"轮数 [{epoch + 1}/{num_epochs}], 损失率: {loss.item():.4f}")

with torch.no_grad():
    test_hidden = None
    test_output,_ = model(inputs.unsqueeze(0),test_hidden)
    pred = torch.argmax(test_output,dim=2).squeeze().numpy()

    print("输入序列:",''.join([idx_to_char[i] for i in input_data]))
    print("预测序列:", ''.join([idx_to_char[i] for i in pred]))

# 模型保存和加载
torch.save(model,'model.pth')
torch.save(model.state_dict(),'model_weights.pth')

# model.load_state_dict(torch.load('model_weights.pth'))
# load_model = torch.load('model.pth')

# 模型检查点保存和加载
checkpoint = {
    'epoch':epoch,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict(),
    'loss':loss,
}
torch.save(checkpoint,'checkout.pth')

checkpoint = torch.load('checkout.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

