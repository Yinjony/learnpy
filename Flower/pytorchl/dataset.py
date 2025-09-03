import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchgen.gen_aoti_c_shim import base_type_to_callsite_expr


class MyDataset(Dataset):
    def __init__(self,X_data,Y_data):
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self,idx):
        x = torch.tensor(self.X_data[idx],dtype=torch.float32)
        y = torch.tensor(self.Y_data[idx],dtype=torch.float32)
        return x,y


X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 输入特征
Y_data = [1, 0, 1, 0]  # 目标标签

dataset = MyDataset(X_data,Y_data)

dataloader = DataLoader(dataset,batch_size=2,shuffle=True)

for epoch in range(1):
    for batch_idx,(inputs,labels) in enumerate(dataloader):
        print(f'Batch {batch_idx + 1}:')
        print(f'Inputs:{inputs}')
        print(f'Labels:{labels}')