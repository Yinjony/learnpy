import torch
from torch.xpu import device

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
tensor = torch.tensor([1,2,3],device=device)

print(tensor.shape)
print(tensor.device)