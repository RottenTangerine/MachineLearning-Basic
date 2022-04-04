
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from icecream import ic
from tqdm import tqdm

train_dataset = torchvision.datasets.CIFAR100(root='../../data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transforms.ToTensor())

train_dataset.class_to_idx

# hyper params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
epoch_num = 5
batch_size = 64
learning_rate = 1e-9
class_num = 100

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class VGG(nn.Module):
    def __init__(self, class_num=100):
        super(VGG, self).__init__()

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # CBR
            else:
                layers += [nn.Conv2d(in_channels, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU(True)]
                in_channels = v

        self.features = nn.Sequential(*layers)
        self.avgPool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, class_num)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

model = VGG().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

torch.cuda.empty_cache()

for epoch in range(epoch_num):
    for i, (pic, label) in enumerate(tqdm(train_loader, leave=False, colour='cyan', desc=f'Epoch {epoch} / {epoch_num}', unit='batches')):
        optimizer.zero_grad()
        pic = pic.to(device)
        label = label.to(device)
        output = model(pic)
        loss = criterion(output, label)
        if (i + 1) % 100 == 0:
            print(f'Epoch: {epoch},batch: {i + 1}, loss: {loss.item()}')
        loss.backward()
        optimizer.step()

model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = model(imgs)

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    print(f'Accruacy: {correct / total}')

