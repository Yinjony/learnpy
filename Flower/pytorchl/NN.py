import torch
import torch.nn as nn
import matplotlib.pyplot as plt

n_in,n_h,n_out,batch_size = 10,5,1,10

x = torch.randn(batch_size,n_in)
y = torch.tensor([[1.0], [0.0], [0.0],
                 [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

model = nn.Sequential(
    nn.Linear(n_in,n_h),
    nn.ReLU(),
    nn.Linear(n_h,n_out),
    nn.Sigmoid()
)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

losses = []

for epoch in range(50):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    losses.append(loss.item())
    print('epoch: ',epoch,'loss: ',loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.figure(figsize=(8,5))
plt.plot(range(1,51),losses,label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.show()

y_pred_final = model(x).detach().numpy()
y_actual = y.numpy()

plt.figure(figsize=(8,5))
plt.plot(range(1,batch_size + 1),y_actual,'o-',label='Actual',color='blue')
plt.plot(range(1,batch_size + 1),y_pred_final,'x--',label='Predicted',color='red')
plt.xlabel('Sample Index')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid()
plt.show()