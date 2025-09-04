import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from PIL import Image

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self,X_data,Y_data):
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, item):
        # 返回指定索引的数据
        x = torch.tensor(self.X_data[item],dtype=torch.float32)
        y = torch.tensor(self.Y_data[item],dtype=torch.float32)
        return x,y

# 示例数据
X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]
Y_data = [1, 0, 1, 0]

# 创建数据集实例
dataset = MyDataset(X_data, Y_data)

# 加载数据，设置每次加载的样本数量
dataloader = DataLoader(dataset,batch_size=2,shuffle=True)

transform = transforms.Compose([
    # 数据预处理
    transforms.Resize((128,128)),#调整大小
    transforms.ToTensor(),#转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #标准化

    # 数据增强
    # transforms.RandomHorizontalFlip(),#随机水平翻转
    # transforms.RandomRotation(30),#随机旋转30度
    # transforms.RandomResizedCrop(128),#随机裁剪并调整为128*128
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('image.jpg')

img_tensor = transform(image)
