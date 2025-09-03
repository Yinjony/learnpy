import torch
from torch.utils.data import Dataset

from pytorchl.RNN import labels


class MyDataset(Dataset):
    def __init__(self,data,labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample,label

data = torch.randn(100,5)
labels = torch.randint(0,2,(100,))

dataset = MyDataset(data,labels)

print("数据集大小:",len(dataset))
print("第0个样本:",dataset[0])