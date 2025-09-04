import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
# 数据加载
train_dataset = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
test_dataset = datasets.MNIST(root='./data',train=False,transform=transform,download=True)

# 替换下载链接
datasets.MNIST.resources = [
    ('https://mirrors.tuna.tsinghua.edu.cn/pytorch-datasets/mnist/train-images-idx3-ubyte.gz',
     'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('https://mirrors.tuna.tsinghua.edu.cn/pytorch-datasets/mnist/train-labels-idx1-ubyte.gz',
     'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://mirrors.tuna.tsinghua.edu.cn/pytorch-datasets/mnist/t10k-images-idx3-ubyte.gz',
     '9fb629c4189551a2d022fa330f9573f3'),
    ('https://mirrors.tuna.tsinghua.edu.cn/pytorch-datasets/mnist/t10k-labels-idx1-ubyte.gz',
     'ec29112dd5afa0611ce80d1b7f02629c')
]

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1,64 * 7 * 7) #展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

# 进行训练
epochs = 5
model.train()

for epoch in range(epochs):
    total_loss = 0
    for imgs,labels in train_loader:
        outputs = model(imgs)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"轮数 [{epoch+1}/{epochs}], 损失率: {total_loss / len(train_loader):.4f}")

# 模型测试，设置为评估模式
model.eval()
correct = 0
total = 0

with torch.no_grad():#关闭梯度计算，没必要因为
    for imgs,labels in test_loader:
        # 把一批图像送进去得到预测结果
        outputs = model(imgs)
        _,pred = torch.max(outputs,1) #取每一行最大值
        total += labels.size(0) #获取这批样本数量
        correct += (pred == labels).sum().item() #累计正确预测的样本数

accuracy = 100 * correct / total
print(f"测试准确率:{accuracy:.2f}%")