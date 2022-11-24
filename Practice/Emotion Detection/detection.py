import matplotlib.pyplot as plt
import numpy as np
import torch
from model import CNN
from torch import nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torch.optim as optim
import os
import time

train_id = int(time.time())
resume_epoch = 0
# Hyper param
epoch_num = 50
batch_size = 16
lr = 1e-4
class_num = 7

# Read data (48x48 pixel gray scale images)
# Dataset link: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer
train_dir = '../../data/emotion-detection/train'
test_dir = '../../data/emotion-detection/test'

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(train_dataset.class_to_idx)
print(test_dataset.class_to_idx)

testdata = next(iter(train_loader))
pic = testdata[0][0]
label = testdata[1][0]

idx_to_class = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
plt.imshow(np.reshape(pic, pic.size()[1:]), cmap='gray')
plt.title(idx_to_class[label])
plt.show()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CNN(class_num).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
torch.cuda.is_available()

scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-8)

loss_list = []
try:
    most_recent_check_point = os.listdir('checkpoint')[-1]
    ckpt_path = os.path.join('checkpoint', most_recent_check_point)
    check_point = torch.load(ckpt_path)
    # load model
    model.load_state_dict(check_point['state_dict'])
    resume_epoch = check_point['epoch']
    loss_list = check_point['loss_list']
    print(f'Successfully load checkpoint {most_recent_check_point}, '
          f'start training from epoch {resume_epoch + 1}')
    plt.plot(loss_list)
    plt.show()
except:
    print('fail to load checkpoint, train from zero beginning')

for _ in range(resume_epoch):
    scheduler.step()

for epoch in range(resume_epoch + 1, epoch_num):
    for i, (pics, labels) in enumerate(train_loader):
        pics = pics.to(device)
        target = torch.tensor(np.eye(class_num)[labels] * 0.8 + 0.1, dtype=torch.float, device=device)
        output = model(pics)
        loss = criterion(output, target)

        if (i + 1) % 500 == 0:
            values, indices = torch.max(output, dim=1)
            correct = (indices.cpu() == labels).sum().item()
            print(f'Epoch: {epoch} / {epoch_num}\t'
                  f'batch: {i + 1} / {len(train_loader)}\t'
                  f'loss: {loss.item():.8f}\t'
                  f'lr: {lr:.8f}\t'
                  f'accuray: {correct} / {batch_size}')
            loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    plt.plot(loss_list)
    plt.show()

    # save ckpt
    if epoch % 5 == 0:
        os.makedirs('checkpoint', exist_ok=True)
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'loss_list': loss_list,
                    }, f'checkpoint/{train_id}_{epoch:03d}.pt')

    # validation
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
        print(f'Validation Accuracy: {correct / total}')

torch.save(model, 'model.pth')

model.state_dict()
