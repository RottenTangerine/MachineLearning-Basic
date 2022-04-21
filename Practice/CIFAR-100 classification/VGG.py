
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

train_dataset = torchvision.datasets.CIFAR100(root='../../data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transforms.ToTensor())


# hyper params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
epoch_num = 50
batch_size = 32
learning_rate = 1e-2
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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

torch.cuda.empty_cache()

for epoch in range(epoch_num):
    for i, (pic, label) in enumerate(train_loader):
        pic = pic.to(device)
        label = label.to(device)

        output = model(pic)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 28 == 0:
            print(f'Epoch: {epoch},batch: {i + 1}, loss: {loss.item():.4f}, lr:{optimizer.param_groups[0]["lr"]:.6f}')

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

