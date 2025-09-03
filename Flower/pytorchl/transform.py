import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

image = Image.open('image.jpg')

image_tensor = transform(image)
print(image_tensor.shape)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transform)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

for inputs,labels in train_loader:
    print(inputs.shape)
    print(labels.shape)

from torch.utils.data import ConcatDataset

dataset1 = Dataset()
dataset2 = Dataset()
combined_dataset = ConcatDataset([dataset1,dataset2])