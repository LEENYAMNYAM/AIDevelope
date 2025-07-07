import os
import random
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from DjangoProject1.settings import MEDIA_ROOT

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1=nn.Conv2d(  #128*128
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        padding=1
    )
    self.conv2=nn.Conv2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=1
    )
    self.conv3=nn.Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        padding=1
    )
    self.conv4=nn.Conv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        padding=1
    )
    self.pool=nn.MaxPool2d(kernel_size=2, stride=2)

    self.fc1=nn.Linear(8*8*64, 128)
    self.fc2=nn.Linear(128, 64)
    self.fc3=nn.Linear(64, 5)

  def forward(self, x): #[3, 128, 128]
    x=self.conv1(x)
    x=torch.relu(x)
    x=self.pool(x)  #[8, 64, 64]
    x=self.conv2(x)
    x=torch.relu(x)
    x=self.pool(x) #[16, 32,32]
    x=self.conv3(x)
    x=torch.relu(x)
    x=self.pool(x) #[32, 16,16]
    x=self.conv4(x)
    x=torch.relu(x)
    x=self.pool(x)  #[64, 8, 8]

    x=x.view(-1, 8*8*64)
    x=self.fc1(x)
    x=self.fc2(x)
    x=self.fc3(x)
    x=torch.log_softmax(x, dim=1)
    return x

def data_pro2(uuid, id,filename):
    torch.manual_seed(321)
    file_path = os.path.join(MEDIA_ROOT, 'posts', str(id), str(uuid))
    save_path=file_path+filename
    img = Image.open(save_path).convert('RGB')
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img)
    print(input_tensor.shape)
    model2 = CNN()
    model_path = os.path.join(MEDIA_ROOT, "model.pt")
    model2.load_state_dict(torch.load(model_path))
    model2.eval()
    with torch.no_grad():
        predict = model2(input_tensor).argmax(dim=1)
        print(predict)
    return predict

def data_pro():
    torch.manual_seed(321)
    # 이미지 크기를 128 x 128 로 조정합니다
    IMAGE_SIZE = 128
    data_path=os.path.join(MEDIA_ROOT,"test_image/")

    test_data = ImageFolder(data_path, transform=transforms.Compose([
                               transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                               transforms.ToTensor()
                             ]))
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

    test_images, labels = next(iter(test_loader))
    # 이미지의 shape을 확인합니다. 128 X 128 RGB 이미지 임을 확인합니다.
    # (batch_size, channel, height, width)
    print(test_images.shape)
    print(labels)

    model2 = CNN()
    model_path=os.path.join(MEDIA_ROOT,"model.pt")
    model2.load_state_dict(torch.load(model_path))
    model2.eval()

    with torch.no_grad():
        predict = model2(test_images).argmax(dim=1)
        print(predict)
    return predict
