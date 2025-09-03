import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np

from pytorchl.CNN import total_loss, accuracy
from pytorchl.NN import criterion
from pytorchl.xianxinghuigui import num_epochs


class SimpleRNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(SimpleRNN,self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out,_ = self.rnn(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return out

num_samples = 1000
seq_len = 10
input_size = 5
output_size = 2

X = torch.randn(num_samples,seq_len,input_size)
Y = torch.randint(0,output_size,(num_samples,))

dataset = TensorDataset(X,Y)
train_loader = DataLoader(dataset,batch_size=32,shuffle=True)

model = SimpleRNN(input_size=input_size,hidden_size=64,output_size=output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs,labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

model.eval()
with torch.no_grad():
    torch = 0
    correct = 0
    for inputs,labels in train_loader:
        outputs = model(inputs)
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")